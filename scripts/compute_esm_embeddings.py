"""
Step 5b (optional): Augment per-atom point clouds with ESM-2 sequence embeddings.

For each pocket-ligand complex:
  1. Re-parse the pocket PDB using the same discard / remap logic as
     compute_pocket_pointclouds.py (so atom order matches exactly).
  2. Walk through atoms in PDB order, grouping consecutive atoms by their
     (chain, residue_sequence_number, residue_name) tuple. This gives a list
     of residues, in order, with their constituent atom indices.
  3. Build a protein sequence using ONLY standard amino acids (the 20 codes).
     Non-protein components (water, metals, cofactors) are skipped from the
     sequence but their atoms still appear in the point cloud.
  4. Run ESM-2 (esm2_t33_650M_UR50D, 1280-dim per residue) on the sequence.
  5. Broadcast each residue's 1280-dim embedding to all its atoms; atoms in
     non-protein residues get a zero vector for the ESM portion.
  6. Load the existing point cloud's positions + 46-dim features, sanity-check
     that the atom ordering matches our re-parse, then concatenate the
     1280-dim ESM features to produce a 1326-dim feature vector per atom.
  7. Save the combined point cloud to data/pocket_pointclouds_esm/<pdb>.npz.

Input:
    data/pocket_pointclouds/<pdb>.npz       (existing 46-dim features)
    data/pdbbind_with_pockets.csv           (manifest with pdb_code, data_subdir)
    ~/Desktop/datasets/PDBbind/P-L/<dir>/<pdb>/<pdb>_pocket.pdb

Output:
    data/pocket_pointclouds_esm/<pdb>.npz   (positions, features=1326-dim,
                                              residue_idx for inspection)

The model trains on the augmented features when config.use_esm_features=True.

Usage:
    python scripts/compute_esm_embeddings.py [--limit N] [--skip-existing]

Dependencies:
    pip install fair-esm torch numpy pandas
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import os
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
# env-overridable for non-PDBbind sources (BioLiP); must match the pointcloud step's dirs
PDBBIND_DIR = Path(os.environ.get("STRUCT_ROOT", "data/structures"))
EXISTING_PC_DIR = Path(os.environ.get("ESM_PC_DIR", str(DATA_DIR / "pocket_pointclouds")))
OUTPUT_DIR = Path(os.environ.get("ESM_OUTPUT_DIR", str(DATA_DIR / "pocket_pointclouds_esm")))
INPUT_CSV = Path(os.environ.get("PC_INPUT_CSV", str(DATA_DIR / "pdbbind_with_pockets.csv")))

# Must match compute_pocket_pointclouds.py exactly so atom order is identical.
DISCARD_ELEMENTS = {"HG", "YB", "EU", "IR", "CS", "SR", "RB", "IN", "GA"}
DISCARD_RESIDUES = {"MSE"}
ELEMENT_REMAP = {"CD": "ZN", "D": "H"}
RESIDUE_REMAP = {"DOD": "HOH"}

# 3-letter to 1-letter amino acid mapping for the 20 standard residues.
AA_3_TO_1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}

# ESM-2 dimensions
ESM_DIM = 1280  # esm2_t33_650M_UR50D
ESM_REPR_LAYER = 33


def extract_element(line):
    """Extract element symbol — same logic as compute_pocket_pointclouds.py:173."""
    element = line[76:78].strip().upper()
    if element:
        return element
    atom_name = line[12:16].strip()
    return "".join(c for c in atom_name if c.isalpha()).upper()[:2]


def parse_pdb_with_residues(pdb_path):
    """Parse a pocket PDB, returning per-atom info AND a per-atom residue index
    into a protein-only sequence (or -1 for non-protein atoms).

    Returns:
        positions:    (N, 3) float array (NOT centroid-centered)
        atom_seq_idx: (N,) int array; -1 for non-standard-amino-acid atoms
        sequence:     str of 1-letter amino acid codes, length = number of
                      standard residues found
    """
    atoms = []  # list of (x, y, z, residue_key, resname)
    if not pdb_path.exists():
        return None

    with open(pdb_path, "r") as f:
        for line in f:
            is_hetatm = line.startswith("HETATM")
            if not line.startswith("ATOM") and not is_hetatm:
                continue
            altloc = line[16]
            if altloc not in (" ", "A", ""):
                continue
            resname = line[17:20].strip()
            if resname in DISCARD_RESIDUES:
                continue
            if resname in RESIDUE_REMAP:
                resname = RESIDUE_REMAP[resname]
            element = extract_element(line)
            element = ELEMENT_REMAP.get(element, element)
            if element in DISCARD_ELEMENTS:
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except (ValueError, IndexError):
                continue
            chain_id = line[21]
            try:
                res_seq = int(line[22:26].strip() or "0")
            except ValueError:
                continue
            i_code = line[26]  # insertion code
            # Residue identity = (chain, res_seq, i_code, resname)
            # is_hetatm matters: same chain+resseq can occur for HETATM vs ATOM.
            residue_key = (chain_id, res_seq, i_code, resname, is_hetatm)
            atoms.append((x, y, z, residue_key, resname))

    if not atoms:
        return None

    # Walk atoms in order, assigning a sequence index to each standard residue.
    # Non-standard residues (HOH, ions, cofactors) get -1 for all their atoms.
    seen_residue_to_seq_idx = {}
    sequence_codes = []
    positions = np.empty((len(atoms), 3), dtype=np.float32)
    atom_seq_idx = np.full(len(atoms), -1, dtype=np.int32)

    for i, (x, y, z, residue_key, resname) in enumerate(atoms):
        positions[i] = (x, y, z)
        if resname in AA_3_TO_1:
            if residue_key not in seen_residue_to_seq_idx:
                seen_residue_to_seq_idx[residue_key] = len(sequence_codes)
                sequence_codes.append(AA_3_TO_1[resname])
            atom_seq_idx[i] = seen_residue_to_seq_idx[residue_key]

    sequence = "".join(sequence_codes)
    return positions, atom_seq_idx, sequence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N pockets (for debugging)")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip pockets that already have an output file")
    parser.add_argument("--max-seq-len", type=int, default=1022,
                        help="Truncate sequences longer than this (ESM limit)")
    parser.add_argument("--relaxed", action="store_true",
                        help="Read pdbbind_with_pockets_relaxed.csv. ESM "
                             ".npz files share the same dir keyed by pdb_code, "
                             "so --skip-existing delta-processes only new codes.")
    args = parser.parse_args()

    input_csv = INPUT_CSV if os.environ.get("PC_INPUT_CSV") else DATA_DIR / (
        "pdbbind_with_pockets_relaxed.csv" if args.relaxed
        else "pdbbind_with_pockets.csv"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Lazy-import esm so the script can still print --help without it.
    import esm

    print(f"Loading ESM-2 (650M, 1280-dim) ...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  Device: {device}")

    df = pd.read_csv(input_csv, low_memory=False)
    print(f"Loaded {len(df)} rows from {input_csv.name}")

    n_done = n_skipped = n_failed = 0
    t_start = time.time()

    iterator = df.itertuples()
    if args.limit:
        iterator = list(iterator)[: args.limit]

    for row in iterator:
        pdb_code = row.pdb_code
        data_subdir = str(row.data_subdir)
        out_path = OUTPUT_DIR / f"{pdb_code}.npz"

        if args.skip_existing and out_path.exists():
            n_skipped += 1
            continue

        # Locate inputs
        pdb_path = PDBBIND_DIR / data_subdir / pdb_code / f"{pdb_code}_pocket.pdb"
        existing_pc_path = EXISTING_PC_DIR / f"{pdb_code}.npz"
        if not pdb_path.exists() or not existing_pc_path.exists():
            n_failed += 1
            continue

        # Re-parse the PDB to extract residues + atom-to-residue mapping
        parsed = parse_pdb_with_residues(pdb_path)
        if parsed is None:
            n_failed += 1
            continue
        new_positions, atom_seq_idx, sequence = parsed

        # Load existing point cloud (positions, features)
        existing = np.load(existing_pc_path)
        existing_positions = existing["positions"]
        existing_features = existing["features"]

        # Sanity check: do our re-parsed positions line up with the existing pc?
        # The existing pc is centroid-centered; ours isn't. Compare AFTER centering.
        if new_positions.shape[0] != existing_features.shape[0]:
            print(f"  [{pdb_code}] atom count mismatch ({new_positions.shape[0]} vs "
                  f"{existing_features.shape[0]}); skipping.")
            n_failed += 1
            continue
        centered_new = new_positions - new_positions.mean(axis=0, keepdims=True)
        if not np.allclose(centered_new, existing_positions, atol=1e-3):
            print(f"  [{pdb_code}] position mismatch after centering; skipping.")
            n_failed += 1
            continue

        # ESM-2 inference (skip if no standard residues found)
        if len(sequence) == 0:
            reps = np.zeros((0, ESM_DIM), dtype=np.float32)
            esm_per_atom = np.zeros((existing_features.shape[0], ESM_DIM), dtype=np.float32)
        else:
            seq = sequence[: args.max_seq_len]  # truncate if needed
            data = [(pdb_code, seq)]
            _, _, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[ESM_REPR_LAYER],
                                return_contacts=False)
            # representations[L] shape: (1, seq_len+2, 1280) — [CLS] and [EOS] flank.
            reps = results["representations"][ESM_REPR_LAYER][0, 1:-1, :].cpu().numpy()  # (L, 1280)

            # Broadcast each residue's embedding to its atoms; zeros for atoms in
            # non-standard residues (-1 sequence index).
            esm_per_atom = np.zeros((existing_features.shape[0], ESM_DIM), dtype=np.float32)
            for i, sidx in enumerate(atom_seq_idx):
                if 0 <= sidx < reps.shape[0]:
                    esm_per_atom[i] = reps[sidx]

        # COMPACT (ESM_COMPACT=1): store per-residue ESM (fp16) + chem (fp16); the
        # dataset loader rebroadcasts to atoms via residue_idx. ~10-16x smaller than
        # the legacy (N,1326)-broadcast format. Legacy path preserved for the
        # existing point clouds (loader auto-detects which format a .npz is).
        if os.environ.get("ESM_COMPACT"):
            np.savez_compressed(
                out_path,
                positions=existing_positions.astype(np.float32),
                chem_features=existing_features.astype(np.float16),
                esm_residue=reps.astype(np.float16),                 # (L, 1280)
                residue_idx=np.asarray(atom_seq_idx, dtype=np.int32),
                sequence_length=np.array(len(sequence), dtype=np.int32),
            )
        else:
            # Concatenate: 46-dim chemistry + 1280-dim ESM = 1326-dim per atom.
            combined_features = np.concatenate([existing_features, esm_per_atom], axis=1).astype(np.float32)
            np.savez(
                out_path,
                positions=existing_positions,
                features=combined_features,
                residue_idx=atom_seq_idx,
                sequence_length=np.array(len(sequence), dtype=np.int32),
            )

        n_done += 1
        if n_done % 25 == 0:
            elapsed = time.time() - t_start
            rate = n_done / elapsed
            remaining = (len(df) - n_done - n_skipped - n_failed) / rate if rate > 0 else 0
            print(f"  done={n_done}  skipped={n_skipped}  failed={n_failed}  "
                  f"elapsed={elapsed/60:.1f} min  ETA={remaining/60:.1f} min")

    print("\nDone.")
    print(f"  Total processed: {n_done}")
    print(f"  Already-existing (skipped): {n_skipped}")
    print(f"  Failed (missing files / parse errors): {n_failed}")
    print(f"  Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
