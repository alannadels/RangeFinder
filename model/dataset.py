"""
Dataset and DataLoader for pocket point clouds.

Each sample is one point cloud (.npz file) paired with 18 target values
(mean and std of 9 ligand properties for that pocket).

Batching uses concatenation with batch indices — no padding, no graphs.
"""

import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class PocketDataset(Dataset):
    """Dataset that loads pocket point clouds and their distribution targets.

    Each item returns:
        positions:  (N, 3) float32 — centroid-centered atom coordinates
        features:   (N, 46) float32 — per-atom chemical features
        targets:    (18,) float32 — normalized distribution targets
        target_mask:(18,) float32 — per-target in-domain flag (1=valid, 0=skip)
        aromatic_present: float32 — 1 if aromatic_rings_mean > 0 else 0
        weight:     float32 — cost-sensitive sample weight
        pdb_code:   str — identifier for the point cloud file
        pocket_id:  str — pocket identifier
        n_atoms:    int — number of atoms in this point cloud

    If the CSV does not contain `in_domain_<target>` columns (i.e., the
    unfiltered original CSVs are loaded), masks default to all-ones and
    aromatic_present is derived from aromatic_rings_mean > 0.
    """

    def __init__(
        self,
        csv_path,
        distributions_csv_path,
        pointcloud_dir,
        target_columns,
        target_mean=None,
        target_std=None,
        sample_weights=None,
        geometric_descriptors_csv_path=None,
        geometric_target_columns=None,
        geometric_target_mean=None,
        geometric_target_std=None,
        # Run 14 exemplar-head plumbing. When provided, the dataset returns
        # per-pocket ligand clouds (variable K x 9) and (mu_true_norm,
        # sigma_true_norm) in the same z-score space the exemplar head's
        # predictions live in.
        ligand_properties_csv_path=None,
        ligand_property_columns=None,
        cloud_mean=None,
        cloud_std=None,
        # Run 15 mixture-head plumbing. When provided, the dataset returns
        # a per-pocket cluster ID per property (hard GMM label) used by
        # HardMixtureLoss.
        cluster_assignments_csv_path=None,
        cluster_property_columns=None,
    ):
        """
        Args:
            csv_path: Path to train.csv, val.csv, or test.csv
            distributions_csv_path: Path to pocket_distributions.csv
            pointcloud_dir: Directory containing {pdb_code}.npz files
            target_columns: List of 18 target column names
            target_mean: (18,) array for normalization (None = no normalization)
            target_std: (18,) array for normalization (None = no normalization)
            sample_weights: (N,) array of per-sample weights (None = uniform)
            geometric_descriptors_csv_path: Path to geometric_descriptors.csv.
                If None, geometric auxiliary targets are disabled (geom_targets
                returned as a zero (8,) tensor; downstream loss must mask).
            geometric_target_columns: Tuple of 8 geometric label column names.
                Required when geometric_descriptors_csv_path is given.
            geometric_target_mean: (8,) array for normalization (None = no norm).
            geometric_target_std: (8,) array for normalization (None = no norm).
        """
        self.pointcloud_dir = Path(pointcloud_dir)
        self.target_columns = target_columns
        self.has_geometric_targets = geometric_descriptors_csv_path is not None
        self.geometric_target_columns = (
            list(geometric_target_columns) if geometric_target_columns else []
        )
        self.has_ligand_clouds = (
            ligand_properties_csv_path is not None
            and ligand_property_columns is not None
            and cloud_mean is not None
            and cloud_std is not None
        )
        self.ligand_property_columns = (
            list(ligand_property_columns) if ligand_property_columns else []
        )
        self.has_cluster_assignments = cluster_assignments_csv_path is not None
        self.cluster_property_columns = (
            list(cluster_property_columns) if cluster_property_columns else []
        )

        # Load CSV and merge with distributions to get targets
        entries = pd.read_csv(csv_path)
        distributions = pd.read_csv(distributions_csv_path)

        # Merge on pocket_id to attach target columns
        merged = entries.merge(
            distributions[["pocket_id"] + target_columns],
            on="pocket_id",
            how="left",
        )

        # Merge geometric labels (per-pocket) on pocket_id if provided.
        if self.has_geometric_targets:
            geom_df = pd.read_csv(geometric_descriptors_csv_path)
            missing = [c for c in self.geometric_target_columns if c not in geom_df.columns]
            if missing:
                raise ValueError(
                    f"geometric_descriptors_csv missing columns: {missing}"
                )
            merged = merged.merge(
                geom_df[["pocket_id"] + self.geometric_target_columns],
                on="pocket_id",
                how="left",
            )

        self.pdb_codes = merged["pdb_code"].values
        self.pocket_ids = merged["pocket_id"].values
        # per-sample cost for dynamic batching = edge count (true GPU-memory driver),
        # falling back to atom count, then None. Used by AtomBudgetBatchSampler.
        if "n_edges" in merged.columns:
            self.atom_counts = merged["n_edges"].values.astype(int)
        elif "n_atoms" in merged.columns:
            self.atom_counts = merged["n_atoms"].values.astype(int)
        else:
            self.atom_counts = None

        # Extract raw targets
        raw_targets = merged[target_columns].values.astype(np.float32)  # (N, 18)

        # Extract per-target in-domain masks if present in the CSV (i.e., when
        # train_filtered.csv etc. are loaded). Otherwise default to all-ones.
        mask_cols = [f"in_domain_{t}" for t in target_columns]
        if all(c in entries.columns for c in mask_cols):
            self.target_masks = entries[mask_cols].values.astype(np.float32)  # (N, 18)
        else:
            self.target_masks = np.ones_like(raw_targets, dtype=np.float32)

        # Aromatic-present binary target. Prefer the column from the filtered
        # CSV; otherwise derive from raw aromatic_rings_mean > 0.
        if "aromatic_present" in entries.columns:
            self.aromatic_present = entries["aromatic_present"].values.astype(np.float32)
        else:
            arom_idx = target_columns.index("aromatic_rings_mean")
            arom_vals = raw_targets[:, arom_idx]
            self.aromatic_present = (arom_vals > 0).astype(np.float32)

        # Handle NaN targets (pockets with <3 ligands may have NaN std)
        # Replace NaN with 0 — these will still be used for training but the
        # cost-sensitive weighting ensures they don't dominate
        nan_mask = np.isnan(raw_targets)
        if nan_mask.any():
            n_nan = nan_mask.sum()
            print(f"Warning: {n_nan} NaN target values found, replacing with 0")
            raw_targets = np.nan_to_num(raw_targets, nan=0.0)

        # Normalize targets
        if target_mean is not None and target_std is not None:
            self.target_mean = target_mean
            self.target_std = target_std
            self.targets = (raw_targets - target_mean) / target_std
        else:
            self.target_mean = None
            self.target_std = None
            self.targets = raw_targets

        self.targets = self.targets.astype(np.float32)

        # Geometric auxiliary targets (8 per sample). When the geometric CSV
        # was not provided, we still emit a zero-padded tensor so the collate
        # layout is stable across configs; the training loop and loss zero
        # out the contribution via use_geometric_targets.
        if self.has_geometric_targets:
            raw_geom = merged[self.geometric_target_columns].values.astype(np.float32)
            # Some pockets have geom labels missing (e.g., degenerate hull); fill
            # with 0 and rely on per-row mask. In practice, all pockets that
            # made it through pocket clustering have valid geom labels.
            geom_nan = np.isnan(raw_geom)
            if geom_nan.any():
                n_nan = geom_nan.sum()
                print(f"Warning: {n_nan} NaN geometric label values found, replacing with 0")
                raw_geom = np.nan_to_num(raw_geom, nan=0.0)
            if geometric_target_mean is not None and geometric_target_std is not None:
                self.geometric_target_mean = geometric_target_mean
                self.geometric_target_std = geometric_target_std
                self.geometric_targets = (raw_geom - geometric_target_mean) / geometric_target_std
            else:
                self.geometric_target_mean = None
                self.geometric_target_std = None
                self.geometric_targets = raw_geom
            self.geometric_targets = self.geometric_targets.astype(np.float32)
        else:
            n_geom = len(self.geometric_target_columns) or 8
            self.geometric_targets = np.zeros((len(self.pdb_codes), n_geom), dtype=np.float32)
            self.geometric_target_mean = None
            self.geometric_target_std = None

        # Run 14: per-pocket ligand clouds + (mu_true_norm, sigma_true_norm)
        # in ligand-z-score space. When use_exemplar_head is on, these feed
        # the pinball + KL + SWD losses.
        n_props = len(self.ligand_property_columns) or 9
        if self.has_ligand_clouds:
            self.cloud_mean = np.asarray(cloud_mean, dtype=np.float32)  # (9,)
            self.cloud_std = np.asarray(cloud_std, dtype=np.float32)    # (9,)

            # Load per-PDB ligand properties and pre-index by pocket_id.
            # We z-score normalize all ligand values up front so the loss
            # sees pre-normalized tensors (cheaper than normalizing in the
            # collate path each batch).
            props = pd.read_csv(ligand_properties_csv_path)
            props_keep = props[["pdb_code"] + list(self.ligand_property_columns)].copy()

            # Map every pdb_code in the dataset to its pocket_id, then
            # gather all ligands per pocket. We use the FULL set of ligands
            # for the pocket (across all PDBs that share the pocket_id),
            # not just the ligand for *this* particular row, because the
            # exemplar predictions are a per-pocket quantity.
            pdb_to_pocket = dict(zip(merged["pdb_code"], merged["pocket_id"]))
            all_pocket_ids_in_split = set(self.pocket_ids)

            # Build pocket_id -> normalized (K, 9) array
            self._ligand_cloud_by_pocket = {}
            raw_props_matrix = props_keep[list(self.ligand_property_columns)].values
            for pid in all_pocket_ids_in_split:
                # Find all pdb_codes mapping to this pocket. The dataset
                # split only contains the PDBs for the current split, but
                # the full ligand cloud for the pocket lives across all
                # PDBs of that pocket regardless of split. Pull from the
                # distributions frame's pdb_codes column.
                row = distributions.loc[distributions["pocket_id"] == pid].iloc[0]
                pdb_str = row.get("pdb_codes", "")
                cloud_codes = [c.strip() for c in str(pdb_str).split(",") if c.strip()]
                cloud_rows = props[props["pdb_code"].isin(cloud_codes)]
                cloud = cloud_rows[list(self.ligand_property_columns)].values.astype(np.float32)
                # Fill any NaN with the per-property mean (becomes 0 after z-score).
                cloud = np.where(np.isnan(cloud), self.cloud_mean[None, :], cloud)
                cloud_norm = (cloud - self.cloud_mean) / self.cloud_std
                self._ligand_cloud_by_pocket[pid] = cloud_norm.astype(np.float32)

            # Per-pocket (mu_true_norm, sigma_true_norm) for the KL loss.
            # mu_true_norm[p] = (raw_mean[p] - cloud_mean[p]) / cloud_std[p]
            # sigma_true_norm[p] = raw_std[p] / cloud_std[p]
            raw_means = merged[[f"{p}_mean" for p in self.ligand_property_columns]].values
            raw_stds = merged[[f"{p}_std" for p in self.ligand_property_columns]].values
            raw_means = np.where(np.isnan(raw_means), self.cloud_mean[None, :], raw_means)
            raw_stds = np.where(np.isnan(raw_stds), 0.0, raw_stds)
            self.mu_true_norm = ((raw_means - self.cloud_mean[None, :])
                                  / self.cloud_std[None, :]).astype(np.float32)
            self.sigma_true_norm = (raw_stds / self.cloud_std[None, :]).astype(np.float32)
        else:
            self.cloud_mean = None
            self.cloud_std = None
            self._ligand_cloud_by_pocket = {}
            self.mu_true_norm = np.zeros((len(self.pdb_codes), n_props), dtype=np.float32)
            self.sigma_true_norm = np.zeros((len(self.pdb_codes), n_props), dtype=np.float32)

        # Run 15: per-pocket cluster assignments (one int per property per
        # sample). 9 columns: {property}_cluster.
        n_props_cluster = len(self.cluster_property_columns) or 9
        if self.has_cluster_assignments:
            ca = pd.read_csv(cluster_assignments_csv_path)
            wanted_cols = [f"{p}_cluster" for p in self.cluster_property_columns]
            ca_keep = ca[["pocket_id"] + wanted_cols].copy()
            ca_lookup = ca_keep.set_index("pocket_id")
            assigns = np.zeros((len(self.pdb_codes), n_props_cluster), dtype=np.int64)
            for i, pid in enumerate(self.pocket_ids):
                if pid in ca_lookup.index:
                    assigns[i] = ca_lookup.loc[pid, wanted_cols].values.astype(np.int64)
                # else: assigns[i] stays at zeros (cluster 0 fallback)
            self.cluster_assignments = assigns
        else:
            self.cluster_assignments = np.zeros(
                (len(self.pdb_codes), n_props_cluster), dtype=np.int64,
            )

        # Sample weights
        if sample_weights is not None:
            self.weights = sample_weights.astype(np.float32)
        else:
            # Uniform weights
            self.weights = np.ones(len(self.pdb_codes), dtype=np.float32)

    def __len__(self):
        return len(self.pdb_codes)

    def __getitem__(self, idx):
        pdb_code = self.pdb_codes[idx]

        # Load point cloud from disk. Two on-disk formats are supported:
        #   legacy : features=(N, 1326) with ESM already broadcast to every atom
        #   compact: chem_features=(N,46, fp16) + esm_residue=(L,1280, fp16) +
        #            residue_idx=(N,); ESM is rebroadcast to atoms here at load time
        #            (~10-16x smaller on disk). Auto-detected per file.
        npz_path = self.pointcloud_dir / f"{pdb_code}.npz"
        data = np.load(npz_path)
        positions = data["positions"].astype(np.float32)   # (N, 3)
        if "esm_residue" in data.files:
            chem = data["chem_features"].astype(np.float32)          # (N, 46)
            esm_res = data["esm_residue"].astype(np.float32)         # (L, 1280)
            ridx = data["residue_idx"]
            esm_atom = np.zeros((chem.shape[0], esm_res.shape[1]), dtype=np.float32)
            valid = (ridx >= 0) & (ridx < esm_res.shape[0])
            esm_atom[valid] = esm_res[ridx[valid]]
            features = np.concatenate([chem, esm_atom], axis=1)      # (N, 1326)
        else:
            features = data["features"].astype(np.float32)           # (N, 1326) legacy

        pocket_id = self.pocket_ids[idx]
        if self.has_ligand_clouds:
            cloud = self._ligand_cloud_by_pocket.get(pocket_id)
            ligand_cloud = (torch.from_numpy(cloud)
                            if cloud is not None and cloud.shape[0] > 0
                            else torch.zeros((1, len(self.ligand_property_columns)),
                                             dtype=torch.float32))
        else:
            ligand_cloud = torch.zeros((1, len(self.ligand_property_columns) or 9),
                                       dtype=torch.float32)

        return {
            "positions": torch.from_numpy(positions),
            "features": torch.from_numpy(features),
            "targets": torch.from_numpy(self.targets[idx]),
            "target_mask": torch.from_numpy(self.target_masks[idx]),
            "aromatic_present": torch.tensor(self.aromatic_present[idx], dtype=torch.float32),
            "weight": torch.tensor(self.weights[idx], dtype=torch.float32),
            "geometric_targets": torch.from_numpy(self.geometric_targets[idx]),
            "ligand_cloud": ligand_cloud,
            "mu_true_norm": torch.from_numpy(self.mu_true_norm[idx]),
            "sigma_true_norm": torch.from_numpy(self.sigma_true_norm[idx]),
            "cluster_assignments": torch.from_numpy(self.cluster_assignments[idx]),
            "pdb_code": pdb_code,
            "pocket_id": pocket_id,
            "n_atoms": positions.shape[0],
            "n_ligands": ligand_cloud.shape[0],
        }


def collate_fn(batch_list):
    """Concatenation batching: stack all atoms into one tensor with batch indices.

    Instead of padding to the max size, we concatenate all point clouds and
    track which atoms belong to which sample via a batch_idx tensor.

    Args:
        batch_list: List of dicts from PocketDataset.__getitem__

    Returns:
        dict with:
            positions:  (total_atoms, 3) float32
            features:   (total_atoms, 46) float32
            batch_idx:  (total_atoms,) long — sample index for each atom
            targets:    (B, 18) float32
            weights:    (B,) float32
            pdb_codes:  list of str
            pocket_ids: list of str
            n_atoms:    list of int
    """
    all_positions = []
    all_features = []
    all_batch_idx = []
    all_targets = []
    all_masks = []
    all_aromatic = []
    all_weights = []
    all_geom = []
    all_clouds = []
    all_lig_batch_idx = []
    all_mu_true = []
    all_sigma_true = []
    all_cluster_assignments = []
    pdb_codes = []
    pocket_ids = []
    n_atoms_list = []
    n_ligands_list = []

    for i, sample in enumerate(batch_list):
        n = sample["n_atoms"]
        k = sample["n_ligands"]
        all_positions.append(sample["positions"])
        all_features.append(sample["features"])
        all_batch_idx.append(torch.full((n,), i, dtype=torch.long))
        all_targets.append(sample["targets"])
        all_masks.append(sample["target_mask"])
        all_aromatic.append(sample["aromatic_present"])
        all_weights.append(sample["weight"])
        all_geom.append(sample["geometric_targets"])
        all_clouds.append(sample["ligand_cloud"])
        all_lig_batch_idx.append(torch.full((k,), i, dtype=torch.long))
        all_mu_true.append(sample["mu_true_norm"])
        all_sigma_true.append(sample["sigma_true_norm"])
        all_cluster_assignments.append(sample["cluster_assignments"])
        pdb_codes.append(sample["pdb_code"])
        pocket_ids.append(sample["pocket_id"])
        n_atoms_list.append(n)
        n_ligands_list.append(k)

    return {
        "positions": torch.cat(all_positions, dim=0),      # (total_atoms, 3)
        "features": torch.cat(all_features, dim=0),        # (total_atoms, 46)
        "batch_idx": torch.cat(all_batch_idx, dim=0),      # (total_atoms,)
        "targets": torch.stack(all_targets, dim=0),         # (B, 18)
        "target_masks": torch.stack(all_masks, dim=0),      # (B, 18)
        "aromatic_present": torch.stack(all_aromatic, dim=0),  # (B,)
        "weights": torch.stack(all_weights, dim=0),         # (B,)
        "geometric_targets": torch.stack(all_geom, dim=0),  # (B, 8)
        # Ragged ligand cloud — Run 14 only
        "ligand_cloud": torch.cat(all_clouds, dim=0),       # (sum_K, 9)
        "ligand_batch_idx": torch.cat(all_lig_batch_idx),   # (sum_K,)
        "mu_true_norm": torch.stack(all_mu_true, dim=0),    # (B, 9)
        "sigma_true_norm": torch.stack(all_sigma_true, dim=0),  # (B, 9)
        "cluster_assignments": torch.stack(all_cluster_assignments, dim=0),  # (B, 9)
        "pdb_codes": pdb_codes,
        "pocket_ids": pocket_ids,
        "n_atoms": n_atoms_list,
        "n_ligands": n_ligands_list,
    }


class AtomBudgetBatchSampler(torch.utils.data.Sampler):
    """Dynamic batching by a total-atom budget instead of a fixed pocket count.

    Pocket sizes vary ~20x here, so a fixed batch count occasionally draws a batch
    of several dense pockets and OOMs the GPU (the SE(3) edge graph explodes). This
    packs each batch up to `atom_budget` atoms -> peak memory is bounded regardless
    of pocket sizes, and NO pockets are dropped (a big one just batches alone).

    Randomness/BatchNorm validity: we SHUFFLE the full index list each epoch, then
    greedily pack in that random order -- so batches are size-coherent (bounded) but
    composition stays random (no size-sorting bias in the running BN stats/gradients).
    """

    def __init__(self, atom_counts, atom_budget, shuffle=True, max_batch=16, seed=42):
        self.atom_counts = [int(a) for a in atom_counts]
        self.atom_budget = int(atom_budget)
        self.shuffle = shuffle
        self.max_batch = max_batch
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        idx = list(range(len(self.atom_counts)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(idx)
            self.epoch += 1
        batch, cur = [], 0
        for i in idx:
            a = self.atom_counts[i]
            if batch and (cur + a > self.atom_budget or len(batch) >= self.max_batch):
                yield batch
                batch, cur = [], 0
            batch.append(i)
            cur += a
        if batch:
            yield batch

    def __len__(self):
        return max(1, sum(self.atom_counts) // self.atom_budget)


def create_dataloader(dataset, batch_size, shuffle=True, num_workers=4, atom_budget=None):
    """Create a DataLoader with our custom collate function. If `atom_budget` is set
    and the dataset exposes per-sample atom counts, use dynamic atom-budget batching
    (bounds GPU memory without dropping large pockets)."""
    if atom_budget and getattr(dataset, "atom_counts", None) is not None:
        sampler = AtomBudgetBatchSampler(dataset.atom_counts, atom_budget, shuffle=shuffle)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
