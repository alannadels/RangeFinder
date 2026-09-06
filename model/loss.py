"""
Cost-sensitive masked MSE + binary cross-entropy loss.

Two components:
  - Regression: per-(sample, target) masked MSE, scaled by per-sample weights.
    Per-target masks come from the applicability domain filter
    (data/train_filtered.csv etc.); when filtering is disabled all masks are
    ones and the loss reduces to the original cost-sensitive MSE.
  - Binary: BCE-with-logits for the aromatic_present head.

The combined loss is loss_reg + bce_weight * loss_bin.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CostSensitiveMSELoss(nn.Module):
    """Weighted masked MSE loss.

    For each (sample, target) pair, computes (pred - target)^2 * mask.
    Then averages over the masked (sample, target) entries, with each
    sample's contribution scaled by its weight.
    """

    def forward(self, predictions, targets, weights, target_mask=None, target_weights=None):
        """
        Args:
            predictions:    (B, 18) predicted values
            targets:        (B, 18) ground truth values
            weights:        (B,) per-sample cost-sensitive weights
            target_mask:    (B, 18) float per-target in-domain mask (1=valid),
                            or None to treat all targets as valid.
            target_weights: (18,) tensor of per-target loss multipliers, or
                            None to weight all targets equally.

        Returns:
            Scalar loss value.
        """
        squared_err = (predictions - targets) ** 2  # (B, 18)

        # Apply per-target weights (broadcasting over the batch dim)
        if target_weights is not None:
            squared_err = squared_err * target_weights.unsqueeze(0)

        if target_mask is None:
            target_mask = torch.ones_like(squared_err)

        masked_err = squared_err * target_mask  # (B, 18)

        # Per-sample mean over *valid* targets. Avoid divide-by-zero by
        # clamping the denominator to at least 1.
        per_sample_valid = target_mask.sum(dim=1).clamp(min=1.0)  # (B,)
        per_sample_mse = masked_err.sum(dim=1) / per_sample_valid  # (B,)

        # Weighted average across the batch. Samples with zero valid targets
        # would contribute nothing; their weight is also down-weighted to
        # avoid skewing the normalization.
        sample_has_valid = (target_mask.sum(dim=1) > 0).float()
        effective_weights = weights * sample_has_valid
        denom = effective_weights.sum().clamp(min=1e-8)
        loss = (effective_weights * per_sample_mse).sum() / denom

        return loss


class AromaticPresentBCELoss(nn.Module):
    """BCE-with-logits for the aromatic_present binary head.

    Applies the same per-sample weights used by the regression loss so that
    the cost-sensitive weighting is consistent across the two heads.
    """

    def forward(self, binary_logit, aromatic_present, weights):
        """
        Args:
            binary_logit:     (B,) raw logits
            aromatic_present: (B,) 0/1 targets
            weights:          (B,) per-sample weights

        Returns:
            Scalar loss value.
        """
        per_sample = F.binary_cross_entropy_with_logits(
            binary_logit, aromatic_present, reduction="none"
        )  # (B,)
        denom = weights.sum().clamp(min=1e-8)
        return (weights * per_sample).sum() / denom


class HardMixtureLoss(nn.Module):
    """Per-property mixture-of-experts loss (Run 15).

    For each property:
      - Cross-entropy on classifier logits vs the GMM hard cluster label
      - MSE on the assigned specialist's prediction:
            pred = cluster_center[k*] + specialist[k*](z)
            loss = MSE(pred, [mu_true, sigma_true])  in normalized space
        Only the ground-truth-cluster specialist receives gradient.

    Per-protein cost-sensitive sample weighting applied to both terms.

    Cluster centers are expected to be stored INSIDE the model's mixture
    heads in the SAME normalized (target-z-score) space as the targets, so
    no per-call conversion is needed. train.py is responsible for
    renormalizing the centers once after model construction.
    """

    def __init__(self, cluster_ce_weight=0.3):
        super().__init__()
        self.cluster_ce_weight = cluster_ce_weight

    def forward(self, cluster_logits, residuals, cluster_centers_per_prop,
                cluster_assignments, target_mean_per_prop, target_std_per_prop,
                weights):
        """
        Args:
            cluster_logits:           dict {prop: (B, K_p)} from model
            residuals:                dict {prop: (B, K_p, 2)} from model
            cluster_centers_per_prop: dict {prop: (K_p, 2)} normalized centers
            cluster_assignments:      dict {prop: (B,) long} hard GMM labels
            target_mean_per_prop:     dict {prop: (B, 2)} normalized
                                       (mu_norm, sigma_norm) ground truth
            target_std_per_prop:      unused for now; kept for API symmetry
            weights:                  (B,) per-sample cost-sensitive weights

        Returns:
            scalar total loss + dict of per-property contributions (for logging)
        """
        device = weights.device
        denom = weights.sum().clamp(min=1e-8)
        total_reg = torch.zeros((), device=device)
        total_ce = torch.zeros((), device=device)
        per_prop_log = {}

        for prop, logits in cluster_logits.items():
            assign = cluster_assignments[prop]                 # (B,)
            res = residuals[prop]                              # (B, K, 2)
            centers = cluster_centers_per_prop[prop]           # (K, 2)
            target_norm = target_mean_per_prop[prop]           # (B, 2)

            B = logits.shape[0]
            batch_idx = torch.arange(B, device=device)
            # Only the ground-truth cluster's specialist is used
            picked_res = res[batch_idx, assign]                # (B, 2)
            picked_center = centers[assign]                    # (B, 2)
            pred = picked_center + picked_res                  # (B, 2)

            sq_err = (pred - target_norm).pow(2).mean(dim=1)   # (B,)
            reg = (weights * sq_err).sum() / denom

            # Classifier CE
            ce_per_sample = F.cross_entropy(
                logits, assign, reduction="none",
            )  # (B,)
            ce = (weights * ce_per_sample).sum() / denom

            total_reg = total_reg + reg
            total_ce = total_ce + ce
            per_prop_log[prop] = {
                "reg": float(reg.item()),
                "ce":  float(ce.item()),
            }

        # Average reg across properties for parity with previous Run 8 loss scale
        n_props = len(cluster_logits)
        total_reg = total_reg / n_props
        total_ce = total_ce / n_props
        total = total_reg + self.cluster_ce_weight * total_ce
        return total, total_reg, total_ce, per_prop_log


class PinballLoss(nn.Module):
    """Multi-quantile pinball (a.k.a. quantile-regression) loss.

    Run 14's exemplar head predicts (B, N_exemplars, 9) values. Sorting along
    the exemplar axis for each property gives marginal quantile predictions
    at tau_i = (i+1)/(N+1) for i=0..N-1 (so for N=9 exemplars: tau in
    {0.1, 0.2, ..., 0.9}).

    For each actual ligand value y in a pocket (per property p), the pinball
    loss at quantile tau_i for that property is:
        L = max(tau_i * (y - q), (tau_i - 1) * (y - q))     where q = q_p_sorted[i]
    The model minimizes this by placing the i-th sorted exemplar so that
    roughly tau_i fraction of true ligands fall below it.

    Loss is computed per pocket (averaging over the pocket's ligands and the
    9*N predicted quantile slots), then a cost-sensitive per-pocket weighted
    average aggregates to a scalar.
    """

    def forward(self, exemplars, ligands, ligand_batch_idx, sample_weights,
                num_samples):
        """
        Args:
            exemplars:        (B, N, 9) predicted exemplar matrix
            ligands:          (sum_K, 9) flat ligand property tensor
            ligand_batch_idx: (sum_K,) which pocket each ligand belongs to
            sample_weights:   (B,) per-pocket cost-sensitive weights
            num_samples:      B (passed explicitly to avoid CUDA/CPU sync)

        Returns:
            Scalar loss.
        """
        B, N, _ = exemplars.shape
        # Sort exemplars per property to obtain marginal quantile predictions
        # (B, N, 9). Sorting is stable so gradients flow back via the values.
        sorted_ex, _ = exemplars.sort(dim=1)

        # tau in (0, 1), open-interval to avoid degenerate boundary quantiles
        device = exemplars.device
        tau = (torch.arange(1, N + 1, device=device, dtype=exemplars.dtype)
               / float(N + 1))  # (N,)

        # For each ligand row, look up its pocket's sorted exemplars and
        # compute pinball loss vs that ligand's 9 property values.
        sorted_per_lig = sorted_ex[ligand_batch_idx]      # (sum_K, N, 9)
        lig_vals = ligands.unsqueeze(1)                   # (sum_K, 1, 9)
        diff = lig_vals - sorted_per_lig                  # (sum_K, N, 9)
        tau_b = tau.view(1, N, 1)
        # max(tau*diff, (tau-1)*diff) is the standard pinball formulation
        pinball = torch.maximum(tau_b * diff, (tau_b - 1.0) * diff)  # (sum_K, N, 9)

        # Per-ligand mean over the N*9 quantile slots
        per_ligand_loss = pinball.mean(dim=[1, 2])  # (sum_K,)

        # Aggregate per pocket via scatter_mean (uniform over the pocket's K ligands)
        from torch_scatter import scatter_mean
        per_pocket_loss = scatter_mean(
            per_ligand_loss, ligand_batch_idx, dim=0, dim_size=num_samples,
        )  # (B,)

        # Cost-sensitive weighted batch average
        denom = sample_weights.sum().clamp(min=1e-8)
        return (sample_weights * per_pocket_loss).sum() / denom


class GaussianKLLoss(nn.Module):
    """KL divergence between predicted Gaussian and ground-truth Gaussian per
    property, averaged across the 9 properties and cost-sensitive weighted
    across the batch.

    Derives mu_hat and sigma_hat as the mean and std of the predicted
    exemplars along the exemplar axis. Closed-form Gaussian KL:

        KL(N(mu_hat, sigma_hat) || N(mu_true, sigma_true)) =
            log(sigma_true / sigma_hat)
            + (sigma_hat^2 + (mu_hat - mu_true)^2) / (2 * sigma_true^2)
            - 1/2

    Couples mean and std learning -- the model cannot get one right without
    paying for the other in the same scalar term.

    Numerical safety: when a pocket's ligands cluster tightly on some
    property (e.g., all 3 have formal_charge=0), sigma_true_norm can be ~0,
    making 1/sigma_true^2 explode to 1e10+ and dominating the entire loss.
    We apply two safeties:
      (1) sigma_true is floored at sigma_floor (default 0.05 in normalized
          space) -- below this threshold the std is essentially "constant"
          and the model shouldn't be penalized for not predicting tighter.
          See loss.py docstring for the per-property interpretation.
      (2) Per-(pocket, property) KL is clipped at kl_clip_max (default 10)
          to prevent any single outlier pocket from dominating gradients.
          Reasonable Gaussian mismatches yield KL in [0, 5]; 10 is permissive
          but kills runaway 1e10 spikes.
    """

    def __init__(self, sigma_floor=0.05, kl_clip_max=10.0):
        super().__init__()
        self.sigma_floor = sigma_floor
        self.kl_clip_max = kl_clip_max

    def forward(self, exemplars, target_mean, target_std, sample_weights):
        """
        Args:
            exemplars:      (B, N, 9) predicted exemplar matrix
            target_mean:    (B, 9) ground truth means (in NORMALIZED ligand
                            space, matching the normalized exemplar outputs)
            target_std:     (B, 9) ground truth stds (same normalization)
            sample_weights: (B,) per-pocket cost-sensitive weights

        Returns:
            Scalar loss.
        """
        # Empirical moments of the predicted exemplars
        mu_hat = exemplars.mean(dim=1)                        # (B, 9)
        # unbiased=False for ML convention (matches torch's default for std
        # estimators in losses); sigma_floor prevents log/divide-by-zero
        sigma_hat = exemplars.std(dim=1, unbiased=False).clamp(min=self.sigma_floor)
        sigma_true = target_std.clamp(min=self.sigma_floor)

        log_ratio = torch.log(sigma_true) - torch.log(sigma_hat)
        term2 = (sigma_hat.pow(2) + (mu_hat - target_mean).pow(2)) / (2 * sigma_true.pow(2))
        kl = log_ratio + term2 - 0.5                          # (B, 9)
        # Cap per-(pocket, property) contribution before averaging so that
        # one bad pocket cannot drive the batch loss to infinity.
        kl = kl.clamp(max=self.kl_clip_max)

        per_pocket_kl = kl.mean(dim=1)                        # (B,)
        denom = sample_weights.sum().clamp(min=1e-8)
        return (sample_weights * per_pocket_kl).sum() / denom


class SlicedWassersteinLoss(nn.Module):
    """Sliced Wasserstein-2 distance between predicted exemplar cloud and
    actual ligand cloud, computed per pocket and cost-sensitive averaged.

    1D Wasserstein-2 has a closed form for empirical distributions: project
    both clouds onto a random unit direction in 9-D, sort both projections,
    sample uniformly at n_grid percentiles via inverse-CDF lookup, compute
    L2. Average over n_proj random directions per loss call (resampled fresh
    each step for gradient diversity, per the design decision).

    Both exemplars and ligands MUST be in the same normalized space before
    this loss is called (z-score using training-set per-property stats).
    Otherwise high-magnitude properties (MW) will dominate the projection
    geometry.
    """

    def __init__(self, n_proj=64, n_grid=100):
        super().__init__()
        self.n_proj = n_proj
        self.n_grid = n_grid

    def forward(self, exemplars, ligands, ligand_batch_idx, sample_weights,
                num_samples):
        """
        Args:
            exemplars:        (B, N, 9) in normalized space
            ligands:          (sum_K, 9) in normalized space
            ligand_batch_idx: (sum_K,)
            sample_weights:   (B,)
            num_samples:      B

        Returns:
            Scalar loss.
        """
        B, N, D = exemplars.shape
        device = exemplars.device

        # Sample n_proj random directions on the unit sphere in 9-D
        theta = torch.randn(self.n_proj, D, device=device, dtype=exemplars.dtype)
        theta = theta / theta.norm(dim=1, keepdim=True)       # (n_proj, 9)

        # Project both clouds: (..., 9) @ (9, n_proj) = (..., n_proj)
        ex_proj = exemplars @ theta.T                          # (B, N, n_proj)
        lig_proj = ligands @ theta.T                           # (sum_K, n_proj)

        # Per-pocket 1D W2 over n_proj projections.
        # Per-pocket loop is fine at batch_size=8 (small B).
        t = (torch.arange(self.n_grid, device=device, dtype=exemplars.dtype) + 0.5) / self.n_grid

        per_pocket = []
        for b in range(B):
            # Sort projected exemplars for this pocket
            ex_sorted, _ = ex_proj[b].sort(dim=0)             # (N, n_proj)
            # Filter ligands for this pocket
            mask = (ligand_batch_idx == b)
            lig_b = lig_proj[mask]                            # (K_b, n_proj)
            K_b = lig_b.shape[0]
            lig_sorted, _ = lig_b.sort(dim=0)                 # (K_b, n_proj)

            # Inverse-CDF lookup at the same n_grid percentiles for both
            idx_ex = (t * N).long().clamp(max=N - 1)          # (n_grid,)
            idx_lig = (t * K_b).long().clamp(max=K_b - 1)
            F_ex = ex_sorted[idx_ex]                          # (n_grid, n_proj)
            F_lig = lig_sorted[idx_lig]
            # Per-projection 1D W2^2; mean over n_proj projections
            w2_sq = ((F_ex - F_lig) ** 2).mean(dim=0)         # (n_proj,)
            per_pocket.append(w2_sq.mean())                   # scalar

        per_pocket = torch.stack(per_pocket)                  # (B,)
        denom = sample_weights.sum().clamp(min=1e-8)
        return (sample_weights * per_pocket).sum() / denom


class GeometricMSELoss(nn.Module):
    """Weighted MSE on the 8 geometric auxiliary targets.

    Same cost-sensitive per-sample weighting as the main regression loss,
    but no per-target masking (all geom labels are valid by construction —
    they come from pure geometry of the .npz / .pdb / .sdf files).
    """

    def forward(self, geom_preds, geom_targets, weights):
        """
        Args:
            geom_preds:   (B, 8) predicted geometric values
            geom_targets: (B, 8) ground truth geometric values
            weights:      (B,) per-sample weights

        Returns:
            Scalar loss value.
        """
        squared_err = (geom_preds - geom_targets) ** 2  # (B, 8)
        per_sample = squared_err.mean(dim=1)            # (B,)
        denom = weights.sum().clamp(min=1e-8)
        return (weights * per_sample).sum() / denom


class GIoURangeLoss(nn.Module):
    """1-D Generalized-IoU loss on the mu +/- k*sigma property ranges.

    Directly optimizes the eval metric (interval overlap) instead of MSE on the
    (mean, std) moments. For each of the 9 properties, both the prediction and
    the ground truth are turned into an interval [mu - k*sigma, mu + k*sigma];
    the loss is (1 - GIoU) of the two intervals.

    Why GIoU (not plain IoU): when the predicted and true intervals are DISJOINT,
    plain IoU is 0 with ZERO gradient, so the model gets no signal to move the
    interval toward the target. GIoU adds a penalty proportional to the gap
    (via the smallest enclosing interval), giving a useful gradient everywhere.
        GIoU = IoU - (C - U) / C
    where U = union length, C = enclosing-interval length.  (1 - GIoU) is 0 for a
    perfect match and grows toward 2 as the intervals move apart.

    IMPORTANT: the model's outputs are in normalized (z-scored) space, and mu and
    sigma of a property are normalized by DIFFERENT stats, so we must denormalize
    both to raw units before forming intervals. GIoU is a dimensionless ratio, so
    property scale (MW ~ hundreds vs fsp3 ~ 0-1) does not need re-weighting.

    Cost-sensitive per-sample weighting (inverse protein frequency) matches the
    other losses; only (sample, property) cells with true sigma > 0 contribute
    (a zero-width true range makes IoU degenerate — same rule as evaluation).
    """

    def __init__(self, k_sigma=2.0, eps=1e-6):
        super().__init__()
        self.k = k_sigma
        self.eps = eps

    def forward(self, reg_preds, targets, target_mean, target_std,
                mean_idx, std_idx, weights):
        """
        Args:
            reg_preds:   (B, 18) predicted targets, normalized space
            targets:     (B, 18) ground-truth targets, normalized space
            target_mean: (18,) per-target normalization mean (tensor, on device)
            target_std:  (18,) per-target normalization std
            mean_idx:    (9,) long — indices of the 9 *_mean targets in the 18
            std_idx:     (9,) long — indices of the 9 *_std targets in the 18
            weights:     (B,) per-sample cost-sensitive weights

        Returns:
            Scalar loss.
        """
        # Denormalize to raw units (mu and sigma have different normalizations)
        pred_raw = reg_preds * target_std + target_mean       # (B, 18)
        true_raw = targets * target_std + target_mean         # (B, 18)

        mu_p = pred_raw[:, mean_idx]                           # (B, 9)
        sd_p = pred_raw[:, std_idx].abs()                     # width must be >= 0
        mu_t = true_raw[:, mean_idx]
        sd_t = true_raw[:, std_idx].abs()

        # Intervals
        plo, phi = mu_p - self.k * sd_p, mu_p + self.k * sd_p
        tlo, thi = mu_t - self.k * sd_t, mu_t + self.k * sd_t

        inter = (torch.minimum(phi, thi) - torch.maximum(plo, tlo)).clamp(min=0.0)
        union = (phi - plo) + (thi - tlo) - inter             # set-union length
        enclose = torch.maximum(phi, thi) - torch.minimum(plo, tlo)

        iou = inter / union.clamp(min=self.eps)
        giou = iou - (enclose - union) / enclose.clamp(min=self.eps)  # (B, 9)
        loss_cell = 1.0 - giou                                 # (B, 9), 0 = perfect

        # Only score cells with a valid (non-degenerate) true range
        mask = (sd_t > self.eps).float()                       # (B, 9)
        per_sample = (loss_cell * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        denom = weights.sum().clamp(min=self.eps)
        return (weights * per_sample).sum() / denom
