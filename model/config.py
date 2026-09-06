"""
Central configuration for the SE(3)-Transformer model.

All hyperparameters live here — no magic numbers scattered in other files.
"""

import dataclasses
from pathlib import Path


@dataclasses.dataclass
class Config:
    # ---- Paths ----
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    pointcloud_dir: Path = None  # set in __post_init__
    train_csv: Path = None
    val_csv: Path = None
    test_csv: Path = None
    pocket_distributions_csv: Path = None
    # Optional: validate against a DIFFERENT distributions file than training. Used when the
    # TRAIN target is a re-weighted/denoised proxy but we want val loss measured against the
    # TRUE ranges (so val loss stays comparable to the from-scratch run). None -> same as train.
    val_pocket_distributions_csv: Path = None
    checkpoint_dir: Path = None

    # ---- Applicability domain filter ----
    # If True, load train_filtered.csv / val_filtered.csv / test_filtered.csv
    # (which include per-target in_domain masks and aromatic_present binary
    # target). If False, load the original CSVs and treat all samples as
    # in-domain for every target.
    use_filtered_data: bool = False

    # ---- Relaxed dataset (Run 12+) ----
    # If True, load train_relaxed.csv / val_relaxed.csv / test_relaxed.csv +
    # pocket_distributions_relaxed.csv. These come from the relaxed PDBbind
    # filter set: resolution <= 3.5 A (was 2.5) and no affinity ceiling
    # (was <= 10 uM exact). Pointcloud .npz directories are shared with the
    # original dataset (keyed by pdb_code), so the only switch is the
    # train/val/test/distributions CSVs.
    # Mutually-exclusive with use_filtered_data at the time of writing —
    # there is no relaxed + filtered variant.
    use_relaxed_data: bool = False

    # Weight on the aromatic_present binary classification loss relative to
    # the regression loss. Only used when filtered data (with aromatic_present
    # column) is loaded.
    aromatic_bce_weight: float = 1.0

    # Per-target regression loss weights. If None, all 18 targets are weighted
    # equally (matches the original behavior). Otherwise must be a tuple of 18
    # floats in the order of target_columns. Used by the CMA-ES loss-weight
    # search to evolve per-target weighting.
    loss_weights: tuple = None

    # ---- ESM-2 sequence features ----
    # If True, load augmented point clouds from data/pocket_pointclouds_esm/
    # whose per-atom features are 46-dim chemistry + 1280-dim ESM-2 embedding
    # (broadcast from the parent residue) = 1326-dim total. Falls back to the
    # original 46-dim point clouds when False.
    use_esm_features: bool = True
    esm_feature_dim: int = 1280  # esm2_t33_650M_UR50D
    # If > 0, project the 1280-dim raw ESM embedding through a learnable
    # Linear(esm_feature_dim, esm_projection_dim) before concatenating with
    # the 46-dim chemistry features. Forces the model to learn a low-rank
    # ESM summary instead of memorizing 1280 raw channels. Set to 0 to use
    # the raw ESM embedding (the original Run 7/8/9 behavior).
    esm_projection_dim: int = 0

    # ---- Target properties ----
    properties: tuple = (
        "MW", "logP", "HBD", "HBA", "TPSA",
        "rotatable_bonds", "formal_charge", "aromatic_rings", "fsp3",
    )
    stats: tuple = ("mean", "std")
    num_targets: int = 18  # len(properties) * len(stats)

    # ---- Geometric auxiliary task (Run 13+) ----
    # If True, load data/geometric_descriptors.csv and add 8 auxiliary
    # geometric targets to the supervision signal. A small head is added
    # to the model that predicts these targets from the same trunk; the
    # loss is `MSE_18reg + aromatic_bce_weight*BCE + geom_loss_weight*MSE_geom`.
    # Idea is that supervising the trunk with pocket shape (volume, SASA,
    # depth, asphericity) and ligand-shape distributions (rgyr mean/std,
    # centroid-offset mean/std) forces it to encode geometry-relevant
    # features that should transfer to the std targets it currently
    # under-predicts.
    use_geometric_targets: bool = False

    # ---- Exemplar / distribution-matching loss head (Run 14+) ----
    # If True, the 18-output regression head is REPLACED with an 81-output
    # exemplar head: 9 synthetic ligands x 9 properties, normalized via the
    # training-set ligand stats. The primary regression loss (MSE on 18
    # (mean, std) targets) is replaced with three distribution-matching
    # losses operating on these exemplars vs the actual K ligands in each
    # pocket:
    #   L_pinball: marginal quantile regression per property
    #   L_KL:      closed-form Gaussian KL between derived (mu_hat,
    #              sigma_hat) and ground-truth (mu_true, sigma_true)
    #   L_SWD:     sliced Wasserstein-2 between predicted exemplar cloud
    #              and actual ligand cloud (z-score normalized, joint
    #              over all 9 properties)
    # The aromatic_present BCE head is unchanged.
    # Default OFF -> Run 8 architecture exactly. Flip to True for Run 14.
    use_exemplar_head: bool = False
    n_exemplars: int = 9       # 9 sorted exemplars give tau in {0.1,...,0.9}
    kl_loss_weight: float = 0.1
    swd_loss_weight: float = 0.5
    swd_n_proj: int = 64       # random projections per SWD evaluation
    swd_n_grid: int = 100      # percentile grid points for 1D W2 inv-CDF

    # ---- Per-property mixture-of-experts heads (Run 15+) ----
    # If True, the single 18-output regression head is REPLACED with per-
    # property mixture heads:
    #   - 9 classifier heads (one per property), each producing K_p logits
    #     over GMM-derived clusters of (mu, sigma) values
    #   - K_p specialist heads per property, each predicting residuals
    #     (Delta_mu, Delta_sigma) from its cluster center
    # At training: hard cluster assignment (from GMM precomputation), only the
    # assigned specialist receives gradient. Classifier trained with CE on
    # GMM hard labels.
    # At inference: argmax classifier, use that specialist + cluster center
    # to produce (mu_hat, sigma_hat) per property.
    # The aromatic_present BCE head is unchanged.
    # Default OFF -> Run 8 architecture exactly. Flip to True for Run 15.
    use_mixture_heads: bool = False
    # Path to GMM cluster definitions (centers, weights, K per property).
    # Generated offline by scripts/fit_property_mixtures.py.
    mixture_json: Path = None  # set in __post_init__
    mixture_assignments_csv: Path = None
    # Width of each per-property head's hidden layer
    mixture_head_hidden: int = 24
    # Weight on the classifier cross-entropy loss term relative to the
    # primary specialist MSE loss.
    mixture_cluster_ce_weight: float = 0.3
    # Names of the 8 geometric label columns in geometric_descriptors.csv.
    # Order is fixed; the head produces outputs in this same order.
    geometric_target_columns: tuple = (
        "pocket_volume",
        "pocket_sasa_approx",
        "pocket_depth_mean",
        "pocket_asphericity",
        "lig_rgyr_mean",
        "lig_rgyr_std",
        "lig_centroid_offset_mean",
        "lig_centroid_offset_std",
    )
    # Weight on the geometric MSE loss term relative to the primary 18-target
    # regression loss. 0.5 keeps the geometric loss meaningful but not
    # dominating. Tune up/down based on per-target geom learning curves.
    geom_loss_weight: float = 0.5
    # Hidden width of the geometric head MLP. Same shape as head_hidden_2.
    geom_head_hidden: int = 48

    # ---- Input ----
    # Base per-atom chem-feature width (before ESM is added in __post_init__).
    # 46 = original structural/chem one-hots (old 1326-d models).
    # 52 = + 6 physical channels (partial charge, Coulombic potential, KD & Eisenberg
    # hydrophobicity, burial, near-metal) -> 1332-d. Pass input_features=52 for the
    # physical-feature runs; the default stays 46 so old checkpoints remain loadable.
    input_features: int = 46

    # ---- Model architecture ----
    hidden_scalars: int = 32       # multiplicity for l=0 (scalar) channels
    hidden_vectors: int = 16       # multiplicity for l=1 (vector) channels
    num_layers: int = 4
    num_heads: int = 4
    lmax: int = 1
    radius_cutoff: float = 5.0     # Angstroms — tunable
    num_radial_basis: int = 16
    radial_mlp_hidden: int = 64

    # ---- Pooling + prediction head ----
    # After pooling: hidden_scalars + hidden_vectors (vector norms) = 48
    head_hidden_1: int = 96
    head_hidden_2: int = 48
    head_dropout: float = 0.5
    trunk_dropout: float = 0.1  # dropout on scalar (l=0) channels in each SE(3) layer
    # Fraction of atoms randomly dropped from each training batch BEFORE the
    # forward pass (DropNode-style data augmentation). Applied only at
    # training time; validation/eval always sees the full pocket. Each epoch
    # the model sees a different random subset of each pocket, so the model
    # cannot memorize specific atom configurations. Set to 0.0 to disable
    # (matches the original Run 7/8 behavior). Typical range: 0.05 to 0.20.
    atom_dropout_rate: float = 0.0

    # ---- Training ----
    batch_size: int = 8
    atom_budget: int = None   # if set, dynamic atom-budget batching (overrides batch_size)
    num_epochs: int = 15
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    lr_patience: int = 5
    lr_factor: float = 0.5
    early_stop_patience: int = 25
    grad_clip_norm: float = 10.0
    num_workers: int = 4
    seed: int = 42

    # ---- IoU range loss ----
    # If True, the regression objective becomes a 1-D Generalized-IoU loss on the
    # mu +/- 2sigma property ranges (directly optimizes the eval metric) instead
    # of MSE on the 18 (mean, std) targets. The aromatic BCE head is unchanged.
    # Per-target loss_weights do NOT apply (IoU is per-property, dimensionless).
    use_iou_loss: bool = False

    # If set (e.g. 3), the sigma (std) target normalization stats are computed
    # over pockets with n_ligands >= this value only, so single/dual-ligand
    # pockets (sigma=0 / unreliable) don't skew the sigma normalization. mu is
    # always normalized over all pockets. Pairs with the per-target loss mask
    # (in_domain_<prop>_std = 0 for low-ligand pockets). None = original behavior.
    sigma_norm_min_ligands: int = None

    # ---- Fine-tuning (init from a pretrained checkpoint) ----
    # If set, model.train loads this .pt as initialization AND reuses its
    # target normalization stats (so the pretrained head stays calibrated).
    init_from_checkpoint: Path = None
    # If True, the init checkpoint holds ONLY the shared SE(3) trunk (e.g. from
    # self-supervised denoising pretraining, scripts/pretrain_denoise.py). Load
    # it with strict=False (property head + target stats are trained fresh).
    init_trunk_only: bool = False
    # Freeze input_embed + this many early SE(3) layers (0 = train everything,
    # num_layers = head-only fine-tuning). Used for low-data fine-tuning.
    freeze_trunk_layers: int = 0

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        if self.pointcloud_dir is None:
            self.pointcloud_dir = self.data_dir / (
                "pocket_pointclouds_esm" if self.use_esm_features else "pocket_pointclouds"
            )
        # When ESM features are active, the per-atom input dim grows by either
        # the raw ESM dim (1280) or the projected ESM dim (esm_projection_dim,
        # default 128) depending on whether projection is enabled.
        if self.use_esm_features:
            esm_dim = (
                self.esm_projection_dim if self.esm_projection_dim > 0
                else self.esm_feature_dim
            )
            self.input_features = self.input_features + esm_dim
        if self.use_filtered_data and self.use_relaxed_data:
            raise ValueError(
                "use_filtered_data and use_relaxed_data cannot both be True — "
                "no relaxed+filtered CSV variant exists."
            )
        if self.use_filtered_data:
            split_suffix = "_filtered"
            dist_suffix = ""
        elif self.use_relaxed_data:
            split_suffix = "_relaxed"
            dist_suffix = "_relaxed"
        else:
            split_suffix = ""
            dist_suffix = ""
        if self.train_csv is None:
            self.train_csv = self.data_dir / f"train{split_suffix}.csv"
        if self.val_csv is None:
            self.val_csv = self.data_dir / f"val{split_suffix}.csv"
        if self.test_csv is None:
            self.test_csv = self.data_dir / f"test{split_suffix}.csv"
        if self.pocket_distributions_csv is None:
            self.pocket_distributions_csv = self.data_dir / f"pocket_distributions{dist_suffix}.csv"
        if self.checkpoint_dir is None:
            self.checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"
        if self.mixture_json is None:
            self.mixture_json = self.data_dir / "property_mixtures.json"
        if self.mixture_assignments_csv is None:
            self.mixture_assignments_csv = self.data_dir / "pocket_cluster_assignments.csv"

    @property
    def target_columns(self):
        """List of 18 target column names in pocket_distributions.csv."""
        return [f"{prop}_{stat}" for prop in self.properties for stat in self.stats]

    @property
    def n_geometric_targets(self):
        """Length of geometric_target_columns — fixed at 8."""
        return len(self.geometric_target_columns)

    @property
    def geometric_descriptors_csv(self):
        """Path to the geometric labels CSV. Same suffix scheme as the
        distributions CSV (no suffix for original, _relaxed for relaxed)."""
        suffix = "_relaxed" if self.use_relaxed_data else ""
        return self.data_dir / f"geometric_descriptors{suffix}.csv"

    @property
    def hidden_irreps_str(self):
        """e3nn irreps string for hidden features."""
        return f"{self.hidden_scalars}x0e + {self.hidden_vectors}x1o"

    @property
    def pooled_dim(self):
        """Dimensionality after invariant pooling."""
        return self.hidden_scalars + self.hidden_vectors
