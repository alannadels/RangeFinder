"""
Step 7: Train the SE(3)-Transformer.

Main entry point for model training. Handles:
    - Data loading with concatenation batching
    - Target normalization (zero mean, unit variance from training set)
    - Cost-sensitive loss (inverse protein frequency weighting)
    - Learning rate scheduling (ReduceLROnPlateau)
    - Early stopping
    - Checkpointing (best model saved)
    - Training metrics logging

Usage:
    python -m model.train

    Or from the project root:
    python model/train.py

Dependencies:
    pip install torch e3nn torch-scatter torch-cluster pandas numpy
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Add project root to path so imports work when run as script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.config import Config
from model.dataset import PocketDataset, create_dataloader
from model.architecture import PocketSE3Transformer
from model.loss import (
    CostSensitiveMSELoss, AromaticPresentBCELoss, GeometricMSELoss,
    PinballLoss, GaussianKLLoss, SlicedWassersteinLoss,
    HardMixtureLoss, GIoURangeLoss,
)
from model.normalization import (
    compute_target_stats, compute_sample_weights, compute_ligand_cloud_stats,
)
from model.utils import seed_everything, save_checkpoint, load_checkpoint, TrainingLogger


def train_one_epoch(model, loader, mse_loss, bce_loss, bce_weight, target_weights,
                    optimizer, device, grad_clip_norm, scaler, epoch,
                    atom_dropout_rate=0.0,
                    geom_loss_fn=None, geom_loss_weight=0.0,
                    # Run 14: exemplar-head loss bundle
                    use_exemplar_head=False,
                    pinball_fn=None, kl_fn=None, swd_fn=None,
                    kl_weight=0.0, swd_weight=0.0,
                    # Run 15: mixture-head loss
                    use_mixture_heads=False,
                    mixture_loss_fn=None,
                    properties=None, mean_idx_tensor=None, std_idx_tensor=None,
                    # IoU range loss (replaces the regression MSE)
                    use_iou_loss=False, iou_loss_fn=None,
                    target_mean_t=None, target_std_t=None):
    """Run one training epoch with mixed precision.

    Returns:
        Average combined training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    total_geom_loss = 0.0
    total_pinball = total_kl = total_swd = 0.0
    total_mix_reg = total_mix_ce = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for batch in pbar:
        positions = batch["positions"].to(device)
        features = batch["features"].to(device)
        batch_idx = batch["batch_idx"].to(device)
        targets = batch["targets"].to(device)
        target_masks = batch["target_masks"].to(device)
        aromatic_present = batch["aromatic_present"].to(device)
        weights = batch["weights"].to(device)
        geom_targets = batch["geometric_targets"].to(device) if geom_loss_fn else None

        # Atom dropout (DropNode): randomly drop a fraction of atoms before
        # the forward pass. Gated by the config flag; when 0.0 the batch is
        # unchanged, exactly reproducing prior runs. Only applied here in the
        # training loop -- validation/eval see the full pocket. Per-sample
        # tensors (targets, masks, aromatic_present, weights) are NOT touched
        # because they are pocket-level, not atom-level.
        if atom_dropout_rate > 0.0:
            keep = torch.rand(positions.shape[0], device=device) > atom_dropout_rate
            positions = positions[keep]
            features = features[keep]
            batch_idx = batch_idx[keep]

        optimizer.zero_grad()

        with autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(positions, features, batch_idx)
            binary_logit = outputs["binary_logit"]
            loss_bin = bce_loss(binary_logit, aromatic_present, weights)

            if use_mixture_heads and "cluster_logits" in outputs:
                # Run 15 path: per-property mixture loss
                cluster_logits = outputs["cluster_logits"]
                residuals = outputs["residuals"]
                # Re-cast residuals to fp32 for stable CE + MSE
                cluster_logits = {p: cl.float() for p, cl in cluster_logits.items()}
                residuals = {p: r.float() for p, r in residuals.items()}
                # Per-property tensor inputs
                cluster_assigns_all = batch["cluster_assignments"].to(device)  # (B, 9)
                cluster_centers_per_prop = {
                    p: model.mixture_heads[p].cluster_centers.float()
                    for p in properties
                }
                target_mean_per_prop = {}
                # For each property p, gather the normalized (mu_norm, sigma_norm)
                # from `targets` using mean_idx_tensor / std_idx_tensor
                for i, p in enumerate(properties):
                    mu_norm = targets[:, mean_idx_tensor[i]]
                    sg_norm = targets[:, std_idx_tensor[i]]
                    target_mean_per_prop[p] = torch.stack([mu_norm, sg_norm], dim=1)  # (B, 2)
                cluster_assignments_per_prop = {
                    p: cluster_assigns_all[:, i].long()
                    for i, p in enumerate(properties)
                }
                mix_total, mix_reg, mix_ce, _per_prop_log = mixture_loss_fn(
                    cluster_logits, residuals, cluster_centers_per_prop,
                    cluster_assignments_per_prop, target_mean_per_prop, None,
                    weights.float(),
                )
                loss = mix_total + bce_weight * loss_bin
                loss_reg = mix_reg
                loss_pin = loss_kl_t = loss_swd_t = None
                loss_geom = None
                loss_mix_ce = mix_ce
            elif use_exemplar_head and "exemplar_preds" in outputs:
                # Run 14 path: pinball + KL + SWD on exemplars
                ex_preds = outputs["exemplar_preds"]  # (B, N, 9)
                # SWD/pinball want fp32 for stable sort + scatter
                ex_preds_fp32 = ex_preds.float()
                lig_cloud = batch["ligand_cloud"].to(device).float()
                lig_batch_idx_t = batch["ligand_batch_idx"].to(device)
                mu_true_norm = batch["mu_true_norm"].to(device).float()
                sigma_true_norm = batch["sigma_true_norm"].to(device).float()
                B = ex_preds.shape[0]

                loss_pin = pinball_fn(ex_preds_fp32, lig_cloud, lig_batch_idx_t,
                                       weights.float(), B)
                loss_kl_t = kl_fn(ex_preds_fp32, mu_true_norm, sigma_true_norm,
                                   weights.float())
                loss_swd_t = swd_fn(ex_preds_fp32, lig_cloud, lig_batch_idx_t,
                                     weights.float(), B)
                loss = loss_pin + kl_weight * loss_kl_t + swd_weight * loss_swd_t \
                       + bce_weight * loss_bin
                loss_reg = loss_pin  # for legacy logging variable
                loss_geom = None
            else:
                reg_preds = outputs["reg_preds"]
                if use_iou_loss:
                    loss_reg = iou_loss_fn(
                        reg_preds.float(), targets, target_mean_t, target_std_t,
                        mean_idx_tensor, std_idx_tensor, weights,
                    )
                else:
                    loss_reg = mse_loss(reg_preds, targets, weights, target_masks, target_weights)
                loss = loss_reg + bce_weight * loss_bin
                loss_pin = loss_kl_t = loss_swd_t = None
                loss_mix_ce = None
                if geom_loss_fn is not None and "geom_preds" in outputs:
                    geom_preds = outputs["geom_preds"]
                    loss_geom = geom_loss_fn(geom_preds, geom_targets, weights)
                    loss = loss + geom_loss_weight * loss_geom
                else:
                    loss_geom = None

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.shape[0]
        total_loss += loss.item() * batch_size
        if loss_geom is not None:
            total_geom_loss += loss_geom.item() * batch_size
        if loss_pin is not None:
            total_pinball += loss_pin.item() * batch_size
            total_kl += loss_kl_t.item() * batch_size
            total_swd += loss_swd_t.item() * batch_size
        if 'loss_mix_ce' in locals() and loss_mix_ce is not None:
            total_mix_reg += loss_reg.item() * batch_size
            total_mix_ce += loss_mix_ce.item() * batch_size
        total_samples += batch_size

        postfix = {
            "loss": f"{loss.item():.4f}",
            "reg":  f"{loss_reg.item():.4f}",
            "bce":  f"{loss_bin.item():.4f}",
        }
        if loss_geom is not None:
            postfix["geom"] = f"{loss_geom.item():.4f}"
        if loss_pin is not None:
            postfix["pin"] = f"{loss_pin.item():.4f}"
            postfix["kl"]  = f"{loss_kl_t.item():.4f}"
            postfix["swd"] = f"{loss_swd_t.item():.4f}"
        if 'loss_mix_ce' in locals() and loss_mix_ce is not None:
            postfix["mix_ce"] = f"{loss_mix_ce.item():.4f}"
        pbar.set_postfix(**postfix)

    avg_loss = total_loss / total_samples
    avg_geom = total_geom_loss / total_samples if geom_loss_fn else 0.0
    avg_pin = total_pinball / total_samples if pinball_fn else 0.0
    avg_kl = total_kl / total_samples if kl_fn else 0.0
    avg_swd = total_swd / total_samples if swd_fn else 0.0
    avg_mix_reg = total_mix_reg / total_samples if use_mixture_heads else 0.0
    avg_mix_ce = total_mix_ce / total_samples if use_mixture_heads else 0.0
    return avg_loss, avg_geom, avg_pin, avg_kl, avg_swd, avg_mix_reg, avg_mix_ce


def derive_18_targets_from_exemplars(exemplars, cloud_mean_t, cloud_std_t,
                                      target_mean_t, target_std_t,
                                      mean_idx, std_idx):
    """Helper: derive (B, 18) predictions in target-normalized space from
    Run 14 exemplars in cloud-normalized space.

    Used by validate() and evaluate() to compute Run 8-comparable metrics.

    Args:
        exemplars:     (B, N, 9)  in cloud-z-score space
        cloud_mean_t:  (9,)       tensor on device
        cloud_std_t:   (9,)
        target_mean_t: (18,)      Run 8 target normalization (tensor on device)
        target_std_t:  (18,)
        mean_idx:      (9,) long  indices of each _mean target in target_columns
        std_idx:       (9,) long  indices of each _std  target in target_columns

    Returns:
        (B, 18) tensor in target-normalized space.
    """
    mu_hat_cloud = exemplars.mean(dim=1)                              # (B, 9)
    sigma_hat_cloud = exemplars.std(dim=1, unbiased=False)            # (B, 9)
    mu_hat_raw = mu_hat_cloud * cloud_std_t + cloud_mean_t            # (B, 9)
    sigma_hat_raw = sigma_hat_cloud * cloud_std_t                     # (B, 9)
    # Renormalize to target space
    mu_hat_tn = (mu_hat_raw - target_mean_t[mean_idx]) / target_std_t[mean_idx]
    sigma_hat_tn = (sigma_hat_raw - target_mean_t[std_idx]) / target_std_t[std_idx]
    B = exemplars.shape[0]
    out = exemplars.new_zeros((B, target_mean_t.shape[0]))
    out[:, mean_idx] = mu_hat_tn
    out[:, std_idx] = sigma_hat_tn
    return out


@torch.no_grad()
def validate(model, loader, device, epoch, bce_weight, target_weights=None,
             collect_geom=False, geom_target_columns=None,
             use_exemplar=False, cloud_mean_t=None, cloud_std_t=None,
             target_mean_t=None, target_std_t=None,
             mean_idx=None, std_idx=None,
             pinball_fn=None, kl_fn=None, swd_fn=None,
             kl_weight=0.0, swd_weight=0.0,
             # Run 15: mixture-mode val: derive (mu_hat, sigma_hat) from the
             # picked specialist + cluster center, compute Run 8-comparable
             # masked MSE + BCE. This is also the train objective for the
             # mixture path so LR/best-checkpoint stay aligned.
             use_mixture=False, properties=None,
             use_iou_loss=False, iou_loss_fn=None):
    """Run validation.

    OFF-path (Run 8/12/13 style): val_loss = masked MSE on 18 targets + BCE.
    Identical formula across all those runs; directly comparable across them.

    ON-path (Run 14, use_exemplar=True): val_loss = same formula as train
    (pinball + kl_weight*KL + swd_weight*SWD + bce_weight*BCE). The Run 8-
    formula derived-MSE is computed alongside as `aux_run8_val_mse` for
    diagnostics; it is logged each epoch but is NOT the val_loss returned
    for checkpoint / LR purposes.

    Returns:
        avg_loss: the val_loss used for best-checkpoint selection + LR.
        aux_run8_val_mse: derived (B, 18) MSE+BCE (Run 8 formula) when
                          use_exemplar=True, else None. Diagnostic only.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_aux_run8 = 0.0  # diagnostic: Run 8-style derived-MSE+BCE val (Run 14 only)
    geom_preds_all = [] if collect_geom else None
    geom_targets_all = [] if collect_geom else None

    pbar = tqdm(loader, desc=f"Epoch {epoch} [val]  ", leave=False)
    for batch in pbar:
        positions = batch["positions"].to(device)
        features = batch["features"].to(device)
        batch_idx = batch["batch_idx"].to(device)
        targets = batch["targets"].to(device)
        target_masks = batch["target_masks"].to(device)
        aromatic_present = batch["aromatic_present"].to(device)

        with autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(positions, features, batch_idx)
            binary_logit = outputs["binary_logit"]
            loss_bin = F.binary_cross_entropy_with_logits(binary_logit, aromatic_present)

            if use_mixture and "cluster_logits" in outputs:
                # Run 15 path: derive (B, 18) predictions from hard mixture
                # (argmax classifier + cluster center + specialist residual)
                B = binary_logit.shape[0]
                reg_preds = torch.zeros((B, len(model.config.target_columns)),
                                        device=device, dtype=binary_logit.dtype)
                for i, prop in enumerate(properties):
                    head = model.mixture_heads[prop]
                    cl = outputs["cluster_logits"][prop]
                    res = outputs["residuals"][prop]
                    k_hat = cl.argmax(dim=-1)                       # (B,)
                    batch_arange = torch.arange(B, device=device)
                    picked_res = res[batch_arange, k_hat]           # (B, 2)
                    picked_center = head.cluster_centers[k_hat]     # (B, 2)
                    pred = picked_center + picked_res               # (B, 2)
                    reg_preds[:, mean_idx[i]] = pred[:, 0]
                    reg_preds[:, std_idx[i]] = pred[:, 1]
                # Now fall through to the SAME val-loss formula as Run 8/12/13
                # (Run 8-comparable scalar; aligns LR+checkpoint with the
                # derived prediction quality)
                sq_err = (reg_preds - targets) ** 2
                if target_weights is not None:
                    sq_err = sq_err * target_weights.unsqueeze(0)
                sq_err = sq_err * target_masks
                per_sample_valid = target_masks.sum(dim=1).clamp(min=1.0)
                per_sample_mse = sq_err.sum(dim=1) / per_sample_valid
                loss_reg = per_sample_mse.mean()
                loss = loss_reg + bce_weight * loss_bin
                aux_run8 = None  # diagnostic only used in exemplar path
            elif use_exemplar and "exemplar_preds" in outputs:
                # ---- Run 14 primary val_loss: train-formula -----------
                ex = outputs["exemplar_preds"].float()
                lig_cloud = batch["ligand_cloud"].to(device).float()
                lig_batch_idx_t = batch["ligand_batch_idx"].to(device)
                mu_true_norm = batch["mu_true_norm"].to(device).float()
                sigma_true_norm = batch["sigma_true_norm"].to(device).float()
                B = ex.shape[0]
                w = torch.ones(B, device=device, dtype=ex.dtype)  # uniform on val
                loss_pin = pinball_fn(ex, lig_cloud, lig_batch_idx_t, w, B)
                loss_kl_t = kl_fn(ex, mu_true_norm, sigma_true_norm, w)
                loss_swd_t = swd_fn(ex, lig_cloud, lig_batch_idx_t, w, B)
                loss = loss_pin + kl_weight * loss_kl_t + swd_weight * loss_swd_t \
                       + bce_weight * loss_bin

                # ---- Run 8-style diagnostic: derived-MSE+BCE ---------
                reg_preds = derive_18_targets_from_exemplars(
                    ex, cloud_mean_t, cloud_std_t, target_mean_t, target_std_t,
                    mean_idx, std_idx,
                )
                sq_err = (reg_preds - targets) ** 2
                if target_weights is not None:
                    sq_err = sq_err * target_weights.unsqueeze(0)
                sq_err = sq_err * target_masks
                per_sample_valid = target_masks.sum(dim=1).clamp(min=1.0)
                per_sample_mse = sq_err.sum(dim=1) / per_sample_valid
                run8_mse = per_sample_mse.mean()
                aux_run8 = (run8_mse + bce_weight * loss_bin).item()
            elif use_iou_loss:
                # IoU range loss: same objective as training (uniform val weights)
                reg_preds = outputs["reg_preds"]
                w = torch.ones(reg_preds.shape[0], device=device, dtype=torch.float32)
                loss_reg = iou_loss_fn(
                    reg_preds.float(), targets, target_mean_t, target_std_t,
                    mean_idx, std_idx, w,
                )
                loss = loss_reg + bce_weight * loss_bin
                aux_run8 = None
            else:
                # OFF-path: Run 8/12/13 formula
                reg_preds = outputs["reg_preds"]
                sq_err = (reg_preds - targets) ** 2
                if target_weights is not None:
                    sq_err = sq_err * target_weights.unsqueeze(0)
                sq_err = sq_err * target_masks
                per_sample_valid = target_masks.sum(dim=1).clamp(min=1.0)
                per_sample_mse = sq_err.sum(dim=1) / per_sample_valid
                loss_reg = per_sample_mse.mean()
                loss = loss_reg + bce_weight * loss_bin
                aux_run8 = None

        if collect_geom and "geom_preds" in outputs:
            geom_preds_all.append(outputs["geom_preds"].float().cpu())
            geom_targets_all.append(batch["geometric_targets"].clone())

        batch_size = targets.shape[0]
        total_loss += loss.item() * batch_size
        if aux_run8 is not None:
            total_aux_run8 += aux_run8 * batch_size
        total_samples += batch_size

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / total_samples
    aux_run8_val_mse = (total_aux_run8 / total_samples) if use_exemplar else None
    if use_exemplar:
        print(f"    [diagnostic] Run 8-style derived-MSE+BCE val: {aux_run8_val_mse:.6f}")

    # Optional: per-target val PCC for the geometric head.
    _ = aux_run8_val_mse  # silence linter; consumed via the print above
    if collect_geom and geom_preds_all:
        gp = torch.cat(geom_preds_all, dim=0).numpy()  # (N, 8)
        gt = torch.cat(geom_targets_all, dim=0).numpy()
        pccs = []
        for i in range(gp.shape[1]):
            # Pearson via cov / (std_p * std_t); skip degenerate targets.
            std_p = gp[:, i].std()
            std_t = gt[:, i].std()
            if std_p < 1e-8 or std_t < 1e-8:
                pccs.append(float("nan"))
                continue
            cov = ((gp[:, i] - gp[:, i].mean()) * (gt[:, i] - gt[:, i].mean())).mean()
            pccs.append(float(cov / (std_p * std_t)))
        col_names = geom_target_columns or [f"geom_{i}" for i in range(len(pccs))]
        pcc_str = " ".join(f"{n}={p:+.3f}" for n, p in zip(col_names, pccs))
        print(f"    geom val PCC: {pcc_str}")

    return avg_loss


def main(config=None):
    if config is None:
        config = Config()
    print(f"Configuration:")
    print(f"  Data dir:        {config.data_dir}")
    print(f"  Radius cutoff:   {config.radius_cutoff} A")
    print(f"  Hidden irreps:   {config.hidden_irreps_str}")
    print(f"  Layers:          {config.num_layers}")
    print(f"  Heads:           {config.num_heads}")
    print(f"  Batch size:      {config.batch_size}")
    print(f"  Learning rate:   {config.learning_rate}")
    print(f"  Targets:         {config.num_targets}")
    print(f"  Use filtered:    {config.use_filtered_data}")
    print(f"  Use relaxed:     {config.use_relaxed_data}")
    print(f"  Use geom:        {getattr(config, 'use_geometric_targets', False)}")
    print(f"  Geom loss wt:    {getattr(config, 'geom_loss_weight', 0.0)}")
    print(f"  Use exemplar:    {getattr(config, 'use_exemplar_head', False)}")
    if getattr(config, 'use_exemplar_head', False):
        print(f"  N exemplars:     {config.n_exemplars}")
        print(f"  KL weight:       {config.kl_loss_weight}")
        print(f"  SWD weight:      {config.swd_loss_weight}")
        print(f"  SWD n_proj:      {config.swd_n_proj}")
    print(f"  Use mixture:     {getattr(config, 'use_mixture_heads', False)}")
    if getattr(config, 'use_mixture_heads', False):
        print(f"  Mixture hidden:  {config.mixture_head_hidden}")
        print(f"  Cluster CE wt:   {config.mixture_cluster_ce_weight}")
    print(f"  Train CSV:       {config.train_csv.name}")
    print(f"  BCE weight:      {config.aromatic_bce_weight}")
    print(f"  Atom dropout:    {getattr(config, 'atom_dropout_rate', 0.0)}")
    print()

    # ---- Seed ----
    seed_everything(config.seed)
    print(f"Random seed: {config.seed}")

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # ---- Target normalization ----
    print("Computing target normalization stats from training set...")
    target_mean, target_std = compute_target_stats(
        config.train_csv, config.pocket_distributions_csv, config.target_columns,
        sigma_min_ligands=getattr(config, "sigma_norm_min_ligands", None)
    )
    print(f"  Target columns: {config.target_columns}")
    print(f"  Mean range: [{target_mean.min():.3f}, {target_mean.max():.3f}]")
    print(f"  Std range:  [{target_std.min():.3f}, {target_std.max():.3f}]")
    print()

    # ---- Fine-tuning: load init checkpoint, reuse ITS normalization ----
    # Reusing the pretrained target_mean/std is essential: the loaded head
    # outputs in the pretrained normalized space, so recomputing normalization
    # from the (small) fine-tune set would silently miscalibrate it.
    init_ckpt = None
    if getattr(config, "init_from_checkpoint", None) is not None:
        print(f"Fine-tuning: loading init checkpoint {config.init_from_checkpoint}")
        init_ckpt = torch.load(
            config.init_from_checkpoint, map_location="cpu", weights_only=False
        )
        if init_ckpt.get("target_mean") is not None:
            target_mean = init_ckpt["target_mean"]
            target_std = init_ckpt["target_std"]
            print("  Reusing pretrained target normalization (keeps the head calibrated).")
        print()

    # ---- Ligand cloud normalization (Run 14+) ----
    use_exemplar = getattr(config, "use_exemplar_head", False)
    if use_exemplar:
        print("Computing ligand cloud (z-score) normalization stats...")
        cloud_mean, cloud_std = compute_ligand_cloud_stats(
            config.train_csv,
            config.data_dir / "pdbbind_with_properties.csv",
            list(config.properties),
        )
        print(f"  Cloud mean range: [{cloud_mean.min():.3f}, {cloud_mean.max():.3f}]")
        print(f"  Cloud std  range: [{cloud_std.min():.3f}, {cloud_std.max():.3f}]")
        print()
    else:
        cloud_mean = cloud_std = None

    # ---- Geometric auxiliary target normalization (Run 13+) ----
    use_geom = getattr(config, "use_geometric_targets", False)
    if use_geom:
        print("Computing geometric target normalization stats...")
        geom_mean, geom_std = compute_target_stats(
            config.train_csv,
            config.geometric_descriptors_csv,
            list(config.geometric_target_columns),
        )
        print(f"  Geom columns: {config.geometric_target_columns}")
        print(f"  Geom mean range: [{geom_mean.min():.3f}, {geom_mean.max():.3f}]")
        print(f"  Geom std range:  [{geom_std.min():.3f}, {geom_std.max():.3f}]")
        print()
    else:
        geom_mean = geom_std = None

    # ---- Sample weights ----
    print("Computing cost-sensitive sample weights...")
    sample_weights = compute_sample_weights(config.train_csv)
    print(f"  Weight range: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")
    print(f"  Weight sum: {sample_weights.sum():.1f} (should ≈ {len(sample_weights)})")
    print()

    # ---- Datasets ----
    use_mixture = getattr(config, "use_mixture_heads", False)
    print("Loading datasets...")
    geom_csv = config.geometric_descriptors_csv if use_geom else None
    geom_cols = config.geometric_target_columns if use_geom else None
    lig_csv = (config.data_dir / "pdbbind_with_properties.csv") if use_exemplar else None
    lig_cols = config.properties if use_exemplar else None
    cluster_csv = config.mixture_assignments_csv if use_mixture else None
    cluster_cols = config.properties if use_mixture else None

    train_dataset = PocketDataset(
        csv_path=config.train_csv,
        distributions_csv_path=config.pocket_distributions_csv,
        pointcloud_dir=config.pointcloud_dir,
        target_columns=config.target_columns,
        target_mean=target_mean,
        target_std=target_std,
        sample_weights=sample_weights,
        geometric_descriptors_csv_path=geom_csv,
        geometric_target_columns=geom_cols,
        geometric_target_mean=geom_mean,
        geometric_target_std=geom_std,
        ligand_properties_csv_path=lig_csv,
        ligand_property_columns=lig_cols,
        cloud_mean=cloud_mean,
        cloud_std=cloud_std,
        cluster_assignments_csv_path=cluster_csv,
        cluster_property_columns=cluster_cols,
    )
    val_dist_csv = getattr(config, "val_pocket_distributions_csv", None) or config.pocket_distributions_csv
    if val_dist_csv != config.pocket_distributions_csv:
        print(f"  Validating against DIFFERENT targets: {Path(val_dist_csv).name} "
              f"(train targets: {Path(config.pocket_distributions_csv).name})")
    val_dataset = PocketDataset(
        csv_path=config.val_csv,
        distributions_csv_path=val_dist_csv,
        pointcloud_dir=config.pointcloud_dir,
        target_columns=config.target_columns,
        target_mean=target_mean,  # Use training stats for validation too
        target_std=target_std,
        sample_weights=None,  # Uniform weights for validation
        geometric_descriptors_csv_path=geom_csv,
        geometric_target_columns=geom_cols,
        geometric_target_mean=geom_mean,
        geometric_target_std=geom_std,
        ligand_properties_csv_path=lig_csv,
        ligand_property_columns=lig_cols,
        cloud_mean=cloud_mean,
        cloud_std=cloud_std,
        cluster_assignments_csv_path=cluster_csv,
        cluster_property_columns=cluster_cols,
    )
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print()

    # ---- DataLoaders ----
    _atom_budget = getattr(config, "atom_budget", None)
    train_loader = create_dataloader(
        train_dataset, config.batch_size, shuffle=True, num_workers=config.num_workers,
        atom_budget=_atom_budget,
    )
    val_loader = create_dataloader(
        val_dataset, config.batch_size, shuffle=False, num_workers=config.num_workers,
        atom_budget=_atom_budget,
    )

    # ---- Model ----
    print("Building model...")
    model = PocketSE3Transformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # ---- Fine-tuning: load pretrained weights + optionally freeze trunk ----
    if init_ckpt is not None:
        if getattr(config, "init_trunk_only", False):
            sd = init_ckpt["model_state_dict"]
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"  Loaded {len(sd)} pretrained TRUNK tensors (strict=False).")
            print(f"    fresh-init (not in pretrain): {len(missing)} tensors "
                  f"(head/binary/etc.) | unexpected: {len(unexpected)}")
            if unexpected:
                print(f"    WARNING unexpected keys: {unexpected[:6]}...")
        else:
            model.load_state_dict(init_ckpt["model_state_dict"])
            print("  Loaded pretrained weights.")
    n_freeze = getattr(config, "freeze_trunk_layers", 0)
    if n_freeze > 0:
        for p in model.input_embed.parameters():
            p.requires_grad = False
        if getattr(model, "esm_projection", None) is not None:
            for p in model.esm_projection.parameters():
                p.requires_grad = False
        for i, layer in enumerate(model.layers):
            if i < n_freeze:
                for p in layer.parameters():
                    p.requires_grad = False
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        print(f"  Froze input_embed + first {n_freeze} of {config.num_layers} SE3 "
              f"layers. Trainable: {n_tr:,} / {n_all:,}")

    # Run 15: renormalize mixture heads' cluster centers from RAW property units
    # into the same target-z-score space the training targets live in. After
    # this, head.cluster_centers[k] = (mu_norm, sigma_norm) per cluster, and
    # the loss can compare `cluster_center[k*] + specialist_residual[k*]`
    # directly against the normalized targets.
    if use_mixture:
        print("Renormalizing cluster centers into target-z-score space...")
        with torch.no_grad():
            for prop, head in model.mixture_heads.items():
                m_idx = config.target_columns.index(f"{prop}_mean")
                s_idx = config.target_columns.index(f"{prop}_std")
                tm = torch.tensor(
                    [target_mean[m_idx], target_mean[s_idx]],
                    dtype=head.cluster_centers.dtype,
                    device=head.cluster_centers.device,
                )
                ts = torch.tensor(
                    [target_std[m_idx], target_std[s_idx]],
                    dtype=head.cluster_centers.dtype,
                    device=head.cluster_centers.device,
                )
                head.cluster_centers.copy_((head.cluster_centers - tm) / ts)
        # Diagnostic: dump first head's normalized centers
        for prop, head in model.mixture_heads.items():
            print(f"    {prop:<20s} K={head.k} normalized centers (first 3):")
            for i in range(min(head.k, 3)):
                c = head.cluster_centers[i].tolist()
                print(f"        cluster {i}: mu_norm={c[0]:+.3f}  sigma_norm={c[1]:+.3f}")
            break  # just show one property
    print()

    # ---- Optimizer, scheduler, loss ----
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_factor,
        patience=config.lr_patience,
    )
    mse_loss = CostSensitiveMSELoss()
    bce_loss = AromaticPresentBCELoss()
    bce_weight = config.aromatic_bce_weight
    geom_loss_fn = GeometricMSELoss() if use_geom else None
    geom_loss_weight = getattr(config, "geom_loss_weight", 0.0) if use_geom else 0.0
    # Run 14 loss bundle
    if use_exemplar:
        pinball_fn = PinballLoss()
        kl_fn = GaussianKLLoss()
        swd_fn = SlicedWassersteinLoss(
            n_proj=config.swd_n_proj, n_grid=config.swd_n_grid,
        )
        kl_weight = config.kl_loss_weight
        swd_weight = config.swd_loss_weight
    else:
        pinball_fn = kl_fn = swd_fn = None
        kl_weight = swd_weight = 0.0
    # Run 15 mixture loss
    mixture_loss_fn = HardMixtureLoss(
        cluster_ce_weight=config.mixture_cluster_ce_weight,
    ) if use_mixture else None
    # IoU range loss (replaces regression MSE when enabled)
    use_iou = getattr(config, "use_iou_loss", False)
    iou_loss_fn = GIoURangeLoss() if use_iou else None
    if use_iou:
        print("Regression objective: 1-D Generalized-IoU on mu +/- 2sigma ranges "
              "(per-target loss_weights ignored).")
    if config.loss_weights is not None:
        target_weights_tensor = torch.tensor(list(config.loss_weights), dtype=torch.float32, device=device)
        print(f"Per-target loss weights: {list(config.loss_weights)}")
    else:
        target_weights_tensor = None

    # Indices of _mean and _std targets in config.target_columns. Used by both
    # the Run 14 exemplar val path AND the Run 15 mixture train/val paths.
    mean_idx_t = torch.tensor(
        [config.target_columns.index(f"{p}_mean") for p in config.properties],
        dtype=torch.long, device=device,
    )
    std_idx_t = torch.tensor(
        [config.target_columns.index(f"{p}_std") for p in config.properties],
        dtype=torch.long, device=device,
    )
    if use_exemplar:
        cloud_mean_t = torch.tensor(cloud_mean, dtype=torch.float32, device=device)
        cloud_std_t = torch.tensor(cloud_std, dtype=torch.float32, device=device)
    else:
        cloud_mean_t = cloud_std_t = None
    if use_exemplar or use_iou:
        target_mean_t = torch.tensor(target_mean, dtype=torch.float32, device=device)
        target_std_t = torch.tensor(target_std, dtype=torch.float32, device=device)
    else:
        target_mean_t = target_std_t = None

    scaler = GradScaler()

    # ---- Logger + unique run ID ----
    # Timestamp + PID guarantees uniqueness even if multiple training jobs
    # start within the same wall-clock second (e.g., parallel evaluations or
    # rapid retries). PIDs are unique per running process on a host.
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    logger = TrainingLogger(config.checkpoint_dir)
    best_checkpoint_path = config.checkpoint_dir / f"se3_pocket_ligand_best_{run_id}.pt"
    print(f"Run ID: {run_id}")
    print(f"Best checkpoint will be saved to: {best_checkpoint_path}")

    # ---- Training loop ----
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_geom, train_pin, train_kl, train_swd, train_mix_reg, train_mix_ce = train_one_epoch(
            model, train_loader, mse_loss, bce_loss, bce_weight, target_weights_tensor,
            optimizer, device, config.grad_clip_norm, scaler, epoch,
            atom_dropout_rate=getattr(config, "atom_dropout_rate", 0.0),
            geom_loss_fn=geom_loss_fn,
            geom_loss_weight=geom_loss_weight,
            use_exemplar_head=use_exemplar,
            pinball_fn=pinball_fn, kl_fn=kl_fn, swd_fn=swd_fn,
            kl_weight=kl_weight, swd_weight=swd_weight,
            use_mixture_heads=use_mixture,
            mixture_loss_fn=mixture_loss_fn,
            properties=list(config.properties) if use_mixture else None,
            mean_idx_tensor=mean_idx_t,
            std_idx_tensor=std_idx_t,
            use_iou_loss=use_iou, iou_loss_fn=iou_loss_fn,
            target_mean_t=target_mean_t, target_std_t=target_std_t,
        )

        # Validate (with per-epoch geom PCC diagnostic when use_geom is on).
        # Run 14/15: val_loss uses the SAME formula as the train objective so
        # that LR and checkpoint selection are aligned. For Run 15 mixture,
        # val_loss is masked MSE on derived (mu_hat, sigma_hat) + BCE
        # (identical formula to Run 8 — directly comparable).
        val_loss = validate(
            model, val_loader, device, epoch, bce_weight, target_weights_tensor,
            collect_geom=use_geom,
            geom_target_columns=list(config.geometric_target_columns) if use_geom else None,
            use_exemplar=use_exemplar,
            cloud_mean_t=cloud_mean_t, cloud_std_t=cloud_std_t,
            target_mean_t=target_mean_t, target_std_t=target_std_t,
            mean_idx=mean_idx_t, std_idx=std_idx_t,
            pinball_fn=pinball_fn, kl_fn=kl_fn, swd_fn=swd_fn,
            kl_weight=kl_weight, swd_weight=swd_weight,
            use_mixture=use_mixture,
            properties=list(config.properties) if use_mixture else None,
            use_iou_loss=use_iou, iou_loss_fn=iou_loss_fn,
        )

        epoch_time = time.time() - epoch_start

        # Learning rate scheduling
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        # Logging
        logger.log_epoch(epoch, train_loss, val_loss, current_lr, best_val_loss)
        geom_str = f" geom_train={train_geom:.6f}" if use_geom else ""
        ex_str = (f" pin={train_pin:.4f} kl={train_kl:.4f} swd={train_swd:.4f}"
                  if use_exemplar else "")
        mix_str = (f" mix_reg={train_mix_reg:.4f} mix_ce={train_mix_ce:.4f}"
                   if use_mixture else "")
        print(f"  Epoch {epoch}: train={train_loss:.6f}{geom_str}{ex_str}{mix_str} "
              f"val={val_loss:.6f} lr={current_lr:.2e} time={epoch_time:.1f}s")

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss, config,
                target_mean, target_std, best_checkpoint_path,
                geom_mean=geom_mean, geom_std=geom_std,
                cloud_mean=cloud_mean, cloud_std=cloud_std,
            )
            logger.log_message(f"New best model saved (val_loss={val_loss:.6f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.early_stop_patience:
            logger.log_message(
                f"Early stopping at epoch {epoch} "
                f"(no improvement for {config.early_stop_patience} epochs)"
            )
            break

    # ---- Done ----
    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best validation loss: {best_val_loss:.6f}")
    print(f"  Checkpoint saved at: {best_checkpoint_path}")
    print(f"  Training log: {logger.csv_path}")


if __name__ == "__main__":
    main()
