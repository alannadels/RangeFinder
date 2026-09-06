"""
Evaluation and inference for the trained SE(3)-Transformer.

Computes per-target and per-property metrics in the original
(unnormalized) target space. Also supports single-sample inference.

Usage:
    python -m model.evaluate
"""

import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.amp import autocast
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.config import Config
from model.dataset import PocketDataset, create_dataloader
from model.architecture import PocketSE3Transformer
from model.utils import load_checkpoint


@torch.no_grad()
def run_inference(model, loader, device, collect_geom=False):
    """Run the model on all batches and collect predictions.

    Returns:
        predictions:      (N, 18) regression predictions (normalized space)
        targets:          (N, 18) ground truth values (normalized space)
        target_masks:     (N, 18) per-target in-domain flags
        binary_logits:    (N,) raw logits from the aromatic_present head
        aromatic_present: (N,) ground truth 0/1
        pdb_codes:        list of str
        pocket_ids:       list of str
        geom_preds:       (N, 8) geometric-head predictions (normalized).
                          Returned only when collect_geom=True; None otherwise.
        geom_targets:     (N, 8) geometric ground truth (normalized).
                          Returned only when collect_geom=True; None otherwise.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_masks = []
    all_logits = []
    all_aromatic = []
    all_pdb_codes = []
    all_pocket_ids = []
    all_geom_preds = [] if collect_geom else None
    all_geom_targets = [] if collect_geom else None

    for batch in tqdm(loader, desc="Inference"):
        positions = batch["positions"].to(device)
        features = batch["features"].to(device)
        batch_idx = batch["batch_idx"].to(device)

        amp_ctx = autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
        with amp_ctx:
            outputs = model(positions, features, batch_idx)

        # The output dict is one of three formats depending on training mode:
        #   Run 8/12/13:  {"reg_preds": (B, 18), "binary_logit": (B,)}
        #   Run 14:       {"exemplar_preds": (B, N, 9), "binary_logit": (B,)}
        #   Run 15:       {"cluster_logits": dict, "residuals": dict, "binary_logit": (B,)}
        # We store whichever is present; main() does the final derivation.
        if "reg_preds" in outputs:
            all_preds.append(outputs["reg_preds"].cpu().numpy())
        elif "exemplar_preds" in outputs:
            all_preds.append(outputs["exemplar_preds"].float().cpu().numpy())
        elif "cluster_logits" in outputs:
            # Stash mixture outputs: for each property, store (cluster_logits,
            # residuals). We use a tuple-keyed sub-dict so the caller can
            # rebuild predictions with cluster centers.
            # Pack into a flat (B, sum_of_K, 3) array? No — variable K per
            # property makes that awkward. Just save as Python objects.
            all_preds.append({
                "cluster_logits": {p: outputs["cluster_logits"][p].cpu().numpy()
                                   for p in outputs["cluster_logits"]},
                "residuals": {p: outputs["residuals"][p].float().cpu().numpy()
                              for p in outputs["residuals"]},
            })
        all_logits.append(outputs["binary_logit"].cpu().numpy())
        all_targets.append(batch["targets"].numpy())
        all_masks.append(batch["target_masks"].numpy())
        all_aromatic.append(batch["aromatic_present"].numpy())
        all_pdb_codes.extend(batch["pdb_codes"])
        all_pocket_ids.extend(batch["pocket_ids"])
        if collect_geom and "geom_preds" in outputs:
            all_geom_preds.append(outputs["geom_preds"].float().cpu().numpy())
            all_geom_targets.append(batch["geometric_targets"].numpy())

    geom_preds_arr = (
        np.concatenate(all_geom_preds, axis=0) if collect_geom and all_geom_preds else None
    )
    geom_targets_arr = (
        np.concatenate(all_geom_targets, axis=0) if collect_geom and all_geom_targets else None
    )
    # For mixture mode all_preds is a list of dicts; preserve as-is for main()
    # to handle. For other modes it's a list of arrays we can concatenate.
    if all_preds and isinstance(all_preds[0], dict):
        # Concatenate batched dicts along axis 0
        merged = {"cluster_logits": {}, "residuals": {}}
        for prop in all_preds[0]["cluster_logits"]:
            merged["cluster_logits"][prop] = np.concatenate(
                [p["cluster_logits"][prop] for p in all_preds], axis=0,
            )
            merged["residuals"][prop] = np.concatenate(
                [p["residuals"][prop] for p in all_preds], axis=0,
            )
        preds_out = merged
    else:
        preds_out = np.concatenate(all_preds, axis=0)
    return (
        preds_out,
        np.concatenate(all_targets, axis=0),
        np.concatenate(all_masks, axis=0),
        np.concatenate(all_logits, axis=0),
        np.concatenate(all_aromatic, axis=0),
        all_pdb_codes,
        all_pocket_ids,
        geom_preds_arr,
        geom_targets_arr,
    )


def denormalize(values, target_mean, target_std):
    """Convert from normalized space back to original units."""
    return values * target_std + target_mean


def compute_metrics(predictions, targets, target_columns, properties, target_masks=None):
    """Compute per-target and per-property evaluation metrics.

    Args:
        predictions:    (N, 18) in original units
        targets:        (N, 18) in original units
        target_columns: list of 18 column names
        properties:     list of 9 property names
        target_masks:   (N, 18) per-target in-domain flags (1 = include).
                        If None, all entries are treated as in-domain.

    Returns:
        dict with:
            per_target: {col_name: {mae, rmse, r2, n}}
            per_property: {prop_name: {mae, rmse}}  (averaged over mean+std)
            overall: {mae, rmse}
    """
    results = {"per_target": {}, "per_property": {}, "overall": {}}

    if target_masks is None:
        target_masks = np.ones_like(predictions)
    mask_bool = target_masks.astype(bool)

    errors = predictions - targets

    # Per-target metrics (over in-domain samples for that target)
    for i, col in enumerate(target_columns):
        m = mask_bool[:, i]
        n = int(m.sum())
        if n == 0:
            results["per_target"][col] = {"mae": float("nan"), "rmse": float("nan"),
                                          "r2": float("nan"), "n": 0}
            continue

        col_errors = errors[m, i]
        col_targets = targets[m, i]

        mae = np.abs(col_errors).mean()
        rmse = np.sqrt((col_errors ** 2).mean())
        ss_res = (col_errors ** 2).sum()
        ss_tot = ((col_targets - col_targets.mean()) ** 2).sum()
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)

        results["per_target"][col] = {"mae": mae, "rmse": rmse, "r2": r2, "n": n}

    # Per-property metrics (average of mean and std targets, masked)
    for prop in properties:
        mean_col = f"{prop}_mean"
        std_col = f"{prop}_std"
        mean_idx = target_columns.index(mean_col)
        std_idx = target_columns.index(std_col)
        m_mean = mask_bool[:, mean_idx]
        m_std = mask_bool[:, std_idx]
        if m_mean.sum() == 0 or m_std.sum() == 0:
            results["per_property"][prop] = {"mae": float("nan"), "rmse": float("nan")}
            continue
        prop_mae = (
            np.abs(errors[m_mean, mean_idx]).mean() +
            np.abs(errors[m_std, std_idx]).mean()
        ) / 2
        prop_rmse = (
            np.sqrt((errors[m_mean, mean_idx] ** 2).mean()) +
            np.sqrt((errors[m_std, std_idx] ** 2).mean())
        ) / 2
        results["per_property"][prop] = {"mae": prop_mae, "rmse": prop_rmse}

    # Overall metrics (over all in-domain (sample, target) pairs)
    overall_errors = errors[mask_bool]
    if overall_errors.size == 0:
        results["overall"]["mae"] = float("nan")
        results["overall"]["rmse"] = float("nan")
    else:
        results["overall"]["mae"] = np.abs(overall_errors).mean()
        results["overall"]["rmse"] = np.sqrt((overall_errors ** 2).mean())

    return results


def print_metrics(results, target_columns):
    """Pretty-print evaluation metrics."""
    print()
    print("=" * 80)
    print("EVALUATION RESULTS (original units)")
    print("=" * 80)

    # Per-target table — includes the in-domain sample count per target
    print(f"\n{'Target':<25} {'MAE':>10} {'RMSE':>10} {'R²':>10} {'n':>8}")
    print("-" * 67)
    for col in target_columns:
        m = results["per_target"][col]
        print(f"{col:<25} {m['mae']:>10.4f} {m['rmse']:>10.4f} {m['r2']:>10.4f} {m['n']:>8d}")

    # Per-property table
    print(f"\n{'Property':<25} {'MAE':>10} {'RMSE':>10}")
    print("-" * 45)
    for prop, m in results["per_property"].items():
        print(f"{prop:<25} {m['mae']:>10.4f} {m['rmse']:>10.4f}")

    # Overall
    print(f"\n{'Overall':<25} {results['overall']['mae']:>10.4f} {results['overall']['rmse']:>10.4f}")


def compute_binary_metrics(binary_logits, aromatic_present):
    """Accuracy / precision / recall / F1 / AUROC for the binary head."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    )
    probs = 1.0 / (1.0 + np.exp(-binary_logits))
    preds = (probs >= 0.5).astype(int)
    truth = aromatic_present.astype(int)
    return {
        "accuracy": accuracy_score(truth, preds),
        "precision": precision_score(truth, preds, zero_division=0),
        "recall": recall_score(truth, preds, zero_division=0),
        "f1": f1_score(truth, preds, zero_division=0),
        "auroc": roc_auc_score(truth, probs) if len(np.unique(truth)) > 1 else float("nan"),
        "n_positive": int(truth.sum()),
        "n_total": int(len(truth)),
    }


def print_binary_metrics(metrics):
    print()
    print("=" * 80)
    print("AROMATIC_PRESENT BINARY HEAD")
    print("=" * 80)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  AUROC:     {metrics['auroc']:.4f}")
    print(f"  Positives: {metrics['n_positive']} / {metrics['n_total']} "
          f"({100.0 * metrics['n_positive'] / metrics['n_total']:.1f}%)")


def compute_geom_metrics(geom_preds, geom_targets, geom_columns):
    """Per-target MAE, RMSE, R², PCC, SCC for the 8 geometric auxiliary outputs.

    Returns:
        dict {col_name: {mae, rmse, r2, pcc, pcc_pval, scc, scc_pval, n}}.
    """
    out = {}
    if geom_preds is None or geom_targets is None:
        for col in geom_columns:
            out[col] = {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"),
                        "pcc": float("nan"), "pcc_pval": float("nan"),
                        "scc": float("nan"), "scc_pval": float("nan"), "n": 0}
        return out
    errors = geom_preds - geom_targets
    for i, col in enumerate(geom_columns):
        y = geom_targets[:, i]
        p = geom_preds[:, i]
        e = errors[:, i]
        mae = float(np.abs(e).mean())
        rmse = float(np.sqrt((e ** 2).mean()))
        ss_res = (e ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = float(1.0 - ss_res / (ss_tot + 1e-8))
        if y.std() < 1e-8 or p.std() < 1e-8:
            pcc, pcc_p, scc, scc_p = 0.0, 1.0, 0.0, 1.0
        else:
            pcc, pcc_p = pearsonr(y, p)
            scc, scc_p = spearmanr(y, p)
            if np.isnan(scc):
                scc, scc_p = 0.0, 1.0
        out[col] = {"mae": mae, "rmse": rmse, "r2": r2,
                    "pcc": float(pcc), "pcc_pval": float(pcc_p),
                    "scc": float(scc), "scc_pval": float(scc_p),
                    "n": int(len(y))}
    return out


def compute_correlations(predictions, targets, target_columns, target_masks=None):
    """Per-target Pearson + Spearman correlations.

    Returns dict {target_name: {pcc, pcc_pval, scc, scc_pval, n}}.
    If target_masks is provided, only in-domain entries per target are used;
    otherwise all rows contribute.
    """
    if target_masks is None:
        target_masks = np.ones_like(predictions)
    mask_bool = target_masks.astype(bool)
    out = {}
    for i, col in enumerate(target_columns):
        m = mask_bool[:, i]
        n = int(m.sum())
        if n < 3:
            out[col] = {"pcc": float("nan"), "pcc_pval": float("nan"),
                        "scc": float("nan"), "scc_pval": float("nan"), "n": n}
            continue
        y = targets[m, i]
        p = predictions[m, i]
        pcc, pcc_p = pearsonr(y, p)
        scc, scc_p = spearmanr(y, p)
        # spearmanr returns nan when an input is constant — treat as no signal
        if np.isnan(scc):
            scc, scc_p = 0.0, 1.0
        out[col] = {"pcc": float(pcc), "pcc_pval": float(pcc_p),
                    "scc": float(scc), "scc_pval": float(scc_p), "n": n}
    return out


def build_config_annotation(config, run_id, checkpoint):
    """Generate the # RUN CONFIGURATION annotation block for the Results files."""
    today = datetime.now().strftime("%m/%d/%Y")
    if config.use_filtered_data:
        train_line = ("data/train_filtered.csv. Per-target +/- 2 sigma in-domain "
                      "masks applied at loss time;\n#                each target trains "
                      "only on its in-domain subset.")
        eval_line = ("data/test_filtered.csv. Per-target metrics computed only on each\n"
                     "#                target's in-domain subset; sample size n varies "
                     "per target (see 'n' column).")
        loss_line = ("Masked MSE over in-domain (sample, target) pairs + "
                     f"BCE for binary aromatic_present\n#                head "
                     f"(weight = {config.aromatic_bce_weight}).")
    else:
        train_line = "data/train.csv (full distribution, no outlier filter)."
        eval_line = ("data/test.csv (full distribution). Per-target metrics computed "
                     "over all test pockets.")
        loss_line = (f"MSE over all 18 targets (equal weights) + BCE for binary "
                     f"aromatic_present head\n#                (weight = "
                     f"{config.aromatic_bce_weight}).")

    loss_weights_desc = "uniform" if config.loss_weights is None else "custom per-target (see config.loss_weights)"
    epoch = checkpoint.get("epoch", "?") if isinstance(checkpoint, dict) else "?"
    val_loss = checkpoint.get("val_loss", float("nan")) if isinstance(checkpoint, dict) else float("nan")

    return (
        "###############################################################################\n"
        f"# RUN CONFIGURATION — Run {run_id}\n"
        "###############################################################################\n"
        f"# Date generated: {today}\n"
        f"# Architecture:   SE(3)-Transformer with regression head (18 outputs) PLUS a\n"
        f"#                 small binary classification head predicting aromatic_present\n"
        f"#                 = 1 if aromatic_rings_mean > 0 else 0. At inference, the\n"
        f"#                 regression outputs for aromatic_rings_mean and\n"
        f"#                 aromatic_rings_std are GATED to 0 when the binary head\n"
        f"#                 predicts no aromatic.\n"
        f"# Training data:  {train_line}\n"
        f"# Evaluation:     {eval_line}\n"
        f"# Geometry:       radius_cutoff = {config.radius_cutoff} A,\n"
        f"#                 num_radial_basis = {config.num_radial_basis},\n"
        f"#                 {config.num_layers} SE(3)-Transformer layers,\n"
        f"#                 hidden irreps {config.hidden_irreps_str},\n"
        f"#                 batch_size = {config.batch_size}.\n"
        f"# Loss:           {loss_line}\n"
        f"#                 Per-target weights: {loss_weights_desc}.\n"
        f"# Training run:   Best epoch = {epoch}\n"
        f"#                 Best val loss = {val_loss:.6f}\n"
        f"# Checkpoint:     model/checkpoints/se3_pocket_ligand_best_{run_id}.pt\n"
        "###############################################################################\n"
        "\n"
    )


def write_results_files(config, run_id, checkpoint, results, correlations,
                        bin_metrics, results_dir, geom_metrics=None):
    """Write Results_<run_id>.txt, Results_<run_id>_PCC.txt, Results_<run_id>_SC.txt."""
    annotation = build_config_annotation(config, run_id, checkpoint)
    has_n_col = bool(config.use_filtered_data)
    n_total = max(c["n"] for c in correlations.values())

    # ---------- Results_<run_id>.txt (MAE / RMSE / R²) ----------
    L = [annotation.rstrip(), "", "=" * 70]
    if has_n_col:
        L.append("EVALUATION RESULTS (original units, per-target in-domain pockets only)")
    else:
        L.append("EVALUATION RESULTS (original units)")
    L.append("=" * 70)
    L.append("")
    if has_n_col:
        L.append(f"{'Target':<25} {'n':>6} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
        L.append("-" * 63)
        for col in config.target_columns:
            m = results["per_target"][col]
            L.append(f"{col:<25} {m['n']:>6} {m['mae']:>10.4f} {m['rmse']:>10.4f} {m['r2']:>10.4f}")
    else:
        L.append(f"{'Target':<25} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
        L.append("-" * 55)
        for col in config.target_columns:
            m = results["per_target"][col]
            L.append(f"{col:<25} {m['mae']:>10.4f} {m['rmse']:>10.4f} {m['r2']:>10.4f}")
    L.append("")
    L.append(f"{'Property':<25} {'MAE':>10} {'RMSE':>10}")
    L.append("-" * 45)
    for prop, m in results["per_property"].items():
        L.append(f"{prop:<25} {m['mae']:>10.4f} {m['rmse']:>10.4f}")
    L.append("")
    L.append(f"{'Overall':<25} {results['overall']['mae']:>10.4f} {results['overall']['rmse']:>10.4f}")
    L.append("")
    L.append("=" * 70)
    L.append("AROMATIC_PRESENT BINARY HEAD")
    L.append("=" * 70)
    L.append(f"  Accuracy:  {bin_metrics['accuracy']:.4f}")
    L.append(f"  Precision: {bin_metrics['precision']:.4f}")
    L.append(f"  Recall:    {bin_metrics['recall']:.4f}")
    L.append(f"  F1:        {bin_metrics['f1']:.4f}")
    L.append(f"  AUROC:     {bin_metrics['auroc']:.4f}")
    L.append(f"  Positives: {bin_metrics['n_positive']} / {bin_metrics['n_total']} "
             f"({100.0 * bin_metrics['n_positive'] / bin_metrics['n_total']:.1f}%)")
    # Geometric auxiliary head — diagnostic section (Run 13+)
    if geom_metrics:
        L.append("")
        L.append("=" * 70)
        L.append("GEOMETRIC AUXILIARY HEAD (8 outputs, diagnostic only)")
        L.append("=" * 70)
        L.append(f"{'Target':<28} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
        L.append("-" * 60)
        for col, m in geom_metrics.items():
            L.append(f"{col:<28} {m['mae']:>10.4f} {m['rmse']:>10.4f} {m['r2']:>10.4f}")
    L.append("")
    L.append(f"Predictions saved to: model/checkpoints/test_predictions_{run_id}.csv")
    (results_dir / f"Results_{run_id}.txt").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---------- helper for correlation files (PCC + SCC share format) ----------
    def write_correlation_file(metric_key, metric_pval_key, metric_short, metric_long,
                                description, suffix):
        L = [annotation.rstrip(), "", "=" * 75]
        L.append(f"{metric_long} ({metric_short}) - Run {run_id} predictions")
        L.append("=" * 75)
        L.append(f"Source CSV:   model/checkpoints/test_predictions_{run_id}.csv")
        L.append(f"Checkpoint:   model/checkpoints/se3_pocket_ligand_best_{run_id}.pt")
        L.append(f"n test pockets: {n_total}")
        L.append("")
        for line in description.splitlines():
            L.append(line)
        L.append("")
        if has_n_col:
            L.append(f"{'Target':<25} {'n':>6} {metric_short:>10}    {'p-value':>12}")
            L.append("-" * 62)
        else:
            L.append(f"{'Target':<25} {metric_short:>10}    {'p-value':>12}")
            L.append("-" * 55)
        means, stds = [], []
        for col in config.target_columns:
            c = correlations[col]
            val = c[metric_key]
            p = c[metric_pval_key]
            p_str = f"{p:.2e}" if p >= 1e-300 else "< 1e-300"
            if has_n_col:
                L.append(f"{col:<25} {c['n']:>6} {val:>+10.4f}    {p_str:>12}")
            else:
                L.append(f"{col:<25} {val:>+10.4f}    {p_str:>12}")
            (means if col.endswith("_mean") else stds).append(val)
        L.append("")
        L.append("-" * (62 if has_n_col else 55))
        L.append(f"{'Mean of _mean targets':<25} {np.mean(means):>+10.4f}")
        L.append(f"{'Mean of _std  targets':<25} {np.mean(stds):>+10.4f}")
        L.append(f"{'Mean across all 18':<25} {np.mean(means + stds):>+10.4f}")

        # Geometric auxiliary head — diagnostic section (Run 13+)
        if geom_metrics:
            L.append("")
            L.append("=" * 75)
            L.append(f"GEOMETRIC AUXILIARY HEAD {metric_short} (8 outputs, diagnostic only)")
            L.append("=" * 75)
            L.append(f"{'Target':<28} {metric_short:>10}    {'p-value':>12}")
            L.append("-" * 58)
            geom_vals = []
            for col, m in geom_metrics.items():
                val = m[metric_key]
                p = m[metric_pval_key]
                p_str = f"{p:.2e}" if p >= 1e-300 else "< 1e-300"
                L.append(f"{col:<28} {val:>+10.4f}    {p_str:>12}")
                geom_vals.append(val)
            L.append("")
            L.append("-" * 58)
            L.append(f"{'Mean across 8 geom targets':<28} {np.mean(geom_vals):>+10.4f}")
        (results_dir / f"Results_{run_id}_{suffix}.txt").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---------- Results_<run_id>_PCC.txt ----------
    write_correlation_file(
        metric_key="pcc",
        metric_pval_key="pcc_pval",
        metric_short="PCC",
        metric_long="PEARSON CORRELATION COEFFICIENT",
        description=("PCC measures linear correlation between y_pred and y_true.\n"
                     "Range: [-1, +1].   Unaffected by scale/offset (unlike R²)."),
        suffix="PCC",
    )

    # ---------- Results_<run_id>_SC.txt ----------
    write_correlation_file(
        metric_key="scc",
        metric_pval_key="scc_pval",
        metric_short="SCC",
        metric_long="SPEARMAN RANK CORRELATION COEFFICIENT",
        description=("SCC measures monotonic correlation between the *ranks* of y_pred and y_true.\n"
                     "Range: [-1, +1].   Robust to outliers and to non-linear distortions like\n"
                     "compression-toward-mean.   Preferred when data is not normally distributed."),
        suffix="SC",
    )


def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint — pick the latest timestamped run unless overridden.
    # File pattern: se3_pocket_ligand_best_YYYYMMDD_HHMMSS.pt
    # Falls back to legacy unsuffixed filename if no timestamped variant exists.
    candidates = sorted(config.checkpoint_dir.glob("se3_pocket_ligand_best_*.pt"))
    legacy = config.checkpoint_dir / "se3_pocket_ligand_best.pt"
    if candidates:
        checkpoint_path = candidates[-1]
        run_id = checkpoint_path.stem.replace("se3_pocket_ligand_best_", "")
    elif legacy.exists():
        checkpoint_path = legacy
        run_id = "legacy"
    else:
        print(f"No checkpoint found in {config.checkpoint_dir}")
        print("Train the model first with: python -m model.train")
        return
    print(f"Run ID: {run_id}")

    print(f"Loading checkpoint: {checkpoint_path}")
    model = PocketSE3Transformer(config).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model)

    target_mean = checkpoint["target_mean"]
    target_std = checkpoint["target_std"]
    print(f"  Trained for {checkpoint['epoch']} epochs")
    print(f"  Best val loss: {checkpoint['val_loss']:.6f}")

    # Exemplar head (Run 14+). Source of truth is the checkpoint -- the live
    # config flag is consulted only as a fallback when the key is missing.
    use_exemplar_ckpt = checkpoint.get("use_exemplar_head", False)
    cloud_mean = checkpoint.get("cloud_mean", None) if use_exemplar_ckpt else None
    cloud_std = checkpoint.get("cloud_std", None) if use_exemplar_ckpt else None
    if use_exemplar_ckpt:
        print(f"  Exemplar head: ENABLED ({checkpoint.get('n_exemplars', 9)} exemplars)")
        # Build derivation indices once
        mean_idx = np.array([
            config.target_columns.index(f"{p}_mean") for p in config.properties
        ])
        std_idx = np.array([
            config.target_columns.index(f"{p}_std") for p in config.properties
        ])

    # Geometric auxiliary head (Run 13+). The checkpoint records whether
    # use_geometric_targets was on at training time and stashes the
    # geom_mean / geom_std normalization stats. We do NOT trust the live
    # config flag here -- the source of truth is the checkpoint.
    use_geom_ckpt = checkpoint.get("use_geometric_targets", False)
    geom_mean = checkpoint.get("geom_mean", None) if use_geom_ckpt else None
    geom_std = checkpoint.get("geom_std", None) if use_geom_ckpt else None
    if use_geom_ckpt:
        print(f"  Geometric head: ENABLED (8 auxiliary targets)")

    # Load test dataset (with geometric labels if the model was trained on them)
    geom_csv = config.geometric_descriptors_csv if use_geom_ckpt else None
    geom_cols = config.geometric_target_columns if use_geom_ckpt else None
    lig_csv = (config.data_dir / "pdbbind_with_properties.csv") if use_exemplar_ckpt else None
    lig_cols = config.properties if use_exemplar_ckpt else None
    test_dataset = PocketDataset(
        csv_path=config.test_csv,
        distributions_csv_path=config.pocket_distributions_csv,
        pointcloud_dir=config.pointcloud_dir,
        target_columns=config.target_columns,
        target_mean=target_mean,
        target_std=target_std,
        sample_weights=None,
        geometric_descriptors_csv_path=geom_csv,
        geometric_target_columns=geom_cols,
        geometric_target_mean=geom_mean,
        geometric_target_std=geom_std,
        ligand_properties_csv_path=lig_csv,
        ligand_property_columns=lig_cols,
        cloud_mean=cloud_mean,
        cloud_std=cloud_std,
        cluster_assignments_csv_path=(
            config.mixture_assignments_csv if checkpoint.get("use_mixture_heads", False) else None
        ),
        cluster_property_columns=(
            config.properties if checkpoint.get("use_mixture_heads", False) else None
        ),
    )
    test_loader = create_dataloader(
        test_dataset, config.batch_size, shuffle=False, num_workers=config.num_workers,
    )
    print(f"Test set: {len(test_dataset)} samples")

    # Run inference
    print("Running inference...")
    (
        preds_norm, targets_norm, target_masks,
        binary_logits, aromatic_present,
        pdb_codes, pocket_ids,
        geom_preds_norm, geom_targets_norm,
    ) = run_inference(model, test_loader, device, collect_geom=use_geom_ckpt)

    # For Run 15 checkpoints, preds_norm is a dict of cluster_logits/residuals.
    # Derive (B, 18) target-normalized predictions by argmax(classifier) +
    # cluster_center + specialist residual.
    use_mixture_ckpt = checkpoint.get("use_mixture_heads", False)
    if use_mixture_ckpt:
        print(f"  Mixture mode: ENABLED (per-property mixture heads)")
        # Load mixture cluster centers from the model's state buffers (already
        # normalized at training time and persisted in the checkpoint state)
        # via the model object.
        # We need the indices of _mean / _std in target_columns.
        mean_idx = np.array([config.target_columns.index(f"{p}_mean")
                              for p in config.properties])
        std_idx = np.array([config.target_columns.index(f"{p}_std")
                             for p in config.properties])
        N = preds_norm["cluster_logits"][config.properties[0]].shape[0]
        derived = np.zeros((N, len(config.target_columns)), dtype=np.float32)
        for i, prop in enumerate(config.properties):
            cl = preds_norm["cluster_logits"][prop]      # (N, K)
            res = preds_norm["residuals"][prop]          # (N, K, 2)
            k_hat = cl.argmax(axis=-1)                   # (N,)
            picked_res = res[np.arange(N), k_hat]        # (N, 2)
            head = model.mixture_heads[prop]
            centers = head.cluster_centers.detach().cpu().numpy()  # (K, 2)
            picked_center = centers[k_hat]               # (N, 2)
            pred = picked_center + picked_res            # (N, 2)
            derived[:, mean_idx[i]] = pred[:, 0]
            derived[:, std_idx[i]] = pred[:, 1]
        preds_norm = derived

    # For Run 14 checkpoints, preds_norm is actually (N, n_exemplars, 9). Derive
    # (mu_hat, sigma_hat) in raw units, then renormalize to target space so
    # the downstream metric pipeline sees the same (N, 18) shape as Run 8.
    if use_exemplar_ckpt:
        exemplars = preds_norm  # (N, n_ex, 9) in cloud-z-score space
        mu_hat_cloud = exemplars.mean(axis=1)             # (N, 9)
        sigma_hat_cloud = exemplars.std(axis=1, ddof=0)   # (N, 9)
        mu_hat_raw = mu_hat_cloud * cloud_std + cloud_mean
        sigma_hat_raw = sigma_hat_cloud * cloud_std
        preds_18_tn = np.zeros((exemplars.shape[0], len(config.target_columns)), dtype=np.float32)
        preds_18_tn[:, mean_idx] = (mu_hat_raw - target_mean[mean_idx]) / target_std[mean_idx]
        preds_18_tn[:, std_idx]  = (sigma_hat_raw - target_mean[std_idx]) / target_std[std_idx]
        preds_norm = preds_18_tn  # back to (N, 18) for downstream code
        print(f"  Exemplar predictions derived to (N, 18): "
              f"mu_hat range raw [{mu_hat_raw.min():.2f}, {mu_hat_raw.max():.2f}], "
              f"sigma_hat range raw [{sigma_hat_raw.min():.2f}, {sigma_hat_raw.max():.2f}]")

    # Denormalize regression targets / predictions
    preds_orig = denormalize(preds_norm, target_mean, target_std)
    targets_orig = denormalize(targets_norm, target_mean, target_std)

    # Gated aromatic prediction: when the binary head says no aromatic rings,
    # override the regression head's aromatic_rings_{mean,std} predictions to 0.
    arom_probs = 1.0 / (1.0 + np.exp(-binary_logits))
    arom_pred_class = (arom_probs >= 0.5).astype(int)
    if "aromatic_rings_mean" in config.target_columns:
        idx_m = config.target_columns.index("aromatic_rings_mean")
        idx_s = config.target_columns.index("aromatic_rings_std")
        gated_preds = preds_orig.copy()
        gated_preds[arom_pred_class == 0, idx_m] = 0.0
        gated_preds[arom_pred_class == 0, idx_s] = 0.0
    else:
        gated_preds = preds_orig

    # Per-target masked regression metrics
    results = compute_metrics(
        gated_preds, targets_orig, config.target_columns, list(config.properties),
        target_masks=target_masks,
    )
    print_metrics(results, config.target_columns)

    # Binary head metrics
    bin_metrics = compute_binary_metrics(binary_logits, aromatic_present)
    print_binary_metrics(bin_metrics)

    # Save predictions to CSV — includes raw regression preds, gated preds,
    # binary logit / prob / class, and per-target in-domain masks for later
    # filtered analysis.
    pred_df = pd.DataFrame(preds_orig, columns=[f"pred_{c}" for c in config.target_columns])
    for i, col in enumerate(config.target_columns):
        pred_df[f"gated_pred_{col}"] = gated_preds[:, i]
        pred_df[f"true_{col}"] = targets_orig[:, i]
        pred_df[f"in_domain_{col}"] = target_masks[:, i].astype(int)
    pred_df["pred_aromatic_present_logit"] = binary_logits
    pred_df["pred_aromatic_present_prob"] = arom_probs
    pred_df["pred_aromatic_present_class"] = arom_pred_class
    pred_df["true_aromatic_present"] = aromatic_present.astype(int)
    pred_df["pdb_code"] = pdb_codes
    pred_df["pocket_id"] = pocket_ids

    pred_path = config.checkpoint_dir / f"test_predictions_{run_id}.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"\nPredictions saved to: {pred_path}")

    # ---------- Auto-write the three Results_<run_id> text files ----------
    correlations = compute_correlations(
        gated_preds, targets_orig, config.target_columns, target_masks,
    )
    # Geometric auxiliary metrics (Run 13+): denormalize and compute the same
    # MAE / RMSE / R² / PCC / SCC battery on the 8 geometric outputs.
    geom_metrics = None
    if use_geom_ckpt and geom_preds_norm is not None:
        geom_preds_orig = denormalize(geom_preds_norm, geom_mean, geom_std)
        geom_targets_orig = denormalize(geom_targets_norm, geom_mean, geom_std)
        geom_metrics = compute_geom_metrics(
            geom_preds_orig, geom_targets_orig,
            list(config.geometric_target_columns),
        )
        print()
        print("=" * 80)
        print("GEOMETRIC AUXILIARY HEAD (8 outputs, diagnostic)")
        print("=" * 80)
        print(f"{'Target':<28} {'MAE':>10} {'RMSE':>10} {'R²':>10} {'PCC':>10} {'SCC':>10}")
        print("-" * 80)
        for col, m in geom_metrics.items():
            print(f"{col:<28} {m['mae']:>10.4f} {m['rmse']:>10.4f} "
                  f"{m['r2']:>10.4f} {m['pcc']:>+10.4f} {m['scc']:>+10.4f}")
    results_dir = PROJECT_ROOT / "data" / "Results"
    results_dir.mkdir(parents=True, exist_ok=True)
    write_results_files(
        config, run_id, checkpoint, results, correlations, bin_metrics, results_dir,
        geom_metrics=geom_metrics,
    )
    print(f"Results files written to: {results_dir}")
    print(f"  Results_{run_id}.txt")
    print(f"  Results_{run_id}_PCC.txt")
    print(f"  Results_{run_id}_SC.txt")


if __name__ == "__main__":
    main()
