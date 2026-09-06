"""
Target normalization and cost-sensitive sample weight computation.

Normalization stats are computed from UNIQUE pockets in the training set
(not per-sample) to avoid bias from proteins with many crystal structures.
"""

import numpy as np
import pandas as pd


def compute_target_stats(train_csv_path, distributions_csv_path, target_columns,
                         sigma_min_ligands=None):
    """Compute mean and std of each target from unique pockets in the training set.

    Uses unique pocket_ids so that a pocket appearing in 50 crystal structures
    doesn't count 50 times toward the normalization statistics.

    Args:
        train_csv_path: Path to train.csv
        distributions_csv_path: Path to pocket_distributions.csv
        target_columns: List of 18 column names (e.g. ["MW_mean", "MW_std", ...])
        sigma_min_ligands: If set (e.g. 3), the std (sigma) target columns are
            normalized over pockets with n_ligands >= this only, so single/dual-
            ligand pockets (whose sigma is 0 / unreliable) don't skew the sigma
            normalization. Mean (mu) columns always use all pockets. None =
            original behavior (all pockets for every column).

    Returns:
        (target_mean, target_std): Each a numpy array of shape (18,).
    """
    train = pd.read_csv(train_csv_path)
    distributions = pd.read_csv(distributions_csv_path)

    # Get unique pocket_ids from training set
    train_pocket_ids = train["pocket_id"].unique()

    # Filter distributions to training pockets only
    train_dists = distributions[distributions["pocket_id"].isin(train_pocket_ids)]

    # Extract target values
    target_values = train_dists[target_columns].values.astype(float)  # (n_pockets, 18)

    # Optionally exclude low-ligand pockets from the SIGMA (std) column stats
    # by NaN-ing them out (nanmean/nanstd then skip them).
    if sigma_min_ligands is not None and "n_ligands" in train_dists.columns:
        low = train_dists["n_ligands"].values < sigma_min_ligands
        std_cols = [i for i, c in enumerate(target_columns) if c.endswith("_std")]
        for i in std_cols:
            target_values[low, i] = np.nan

    target_mean = np.nanmean(target_values, axis=0).astype(np.float32)
    target_std = np.nanstd(target_values, axis=0).astype(np.float32)

    # Prevent division by zero — if a target has zero std, set to 1.0
    target_std[target_std < 1e-8] = 1.0

    return target_mean, target_std


def compute_ligand_cloud_stats(train_csv_path, properties_csv_path, property_columns):
    """Compute per-property mean / std from training-set ligands only.

    Run 14 needs per-ligand (not per-pocket) property values for the pinball,
    KL, and Sliced-Wasserstein losses, which compare the predicted exemplar
    cloud to the actual K ligands in each pocket. Both the predicted and
    actual clouds are z-score normalized using these training-set stats so
    that the loss isn't dominated by high-magnitude properties (MW ~ hundreds)
    over low-magnitude ones (fsp3 ~ 0-1).

    Stats are computed over training-set PDB ligands only (not pocket
    aggregates) so that a pocket with many ligands doesn't get up-weighted
    in normalization. A pocket with 50 ligands contributes 50 samples to
    the stats, but that's the right thing for per-ligand normalization --
    we want the stats to match the population of ligand values the SWD
    loss will see at training time.

    Args:
        train_csv_path: Path to train.csv (defines the training pdb_codes)
        properties_csv_path: Path to pdbbind_with_properties.csv
        property_columns: 9 property column names

    Returns:
        (cloud_mean, cloud_std): each a (9,) numpy float32 array.
    """
    train = pd.read_csv(train_csv_path)
    props = pd.read_csv(properties_csv_path)

    # Only training PDBs contribute to the stats
    train_pdb_codes = set(train["pdb_code"].unique())
    train_props = props[props["pdb_code"].isin(train_pdb_codes)]

    values = train_props[list(property_columns)].values  # (N_ligands, 9)
    cloud_mean = np.nanmean(values, axis=0).astype(np.float32)
    cloud_std = np.nanstd(values, axis=0).astype(np.float32)

    # Prevent division by zero (formal_charge often has near-zero std on a
    # small subset). 1.0 keeps the normalized value at its raw magnitude.
    cloud_std[cloud_std < 1e-8] = 1.0

    return cloud_mean, cloud_std


def compute_sample_weights(train_csv_path):
    """Compute cost-sensitive weights inversely proportional to protein frequency.

    Each sample's weight = N_total / (N_proteins * count_of_its_protein).
    This ensures each protein contributes equally to the total loss regardless
    of how many crystal structures it has.

    Weights sum to N_total (preserving loss magnitude).

    Args:
        train_csv_path: Path to train.csv

    Returns:
        weights: numpy array of shape (N_samples,), same order as rows in train.csv.
    """
    train = pd.read_csv(train_csv_path)

    # Count how many samples each protein has
    protein_counts = train["uniprot_id"].value_counts()

    n_total = len(train)
    n_proteins = train["uniprot_id"].nunique()

    # Assign weight to each sample
    weights = train["uniprot_id"].map(
        lambda uid: n_total / (n_proteins * protein_counts[uid])
    ).values.astype(np.float32)

    return weights
