"""
Step 5: Convert pocket PDB files into point clouds for the SE(3)-Transformer.

This script reads the pocket-assigned dataset (pdbbind_with_pockets.csv) and
for each entry, parses the corresponding PDBbind pocket PDB file into a
centroid-centered point cloud with per-atom chemical features.

Input:
    ../data/pdbbind_with_pockets.csv  (produced by compute_pocket_distributions.py)

    For each entry, the pocket structure file is loaded from:
    ~/Desktop/datasets/PDBbind/P-L/{data_subdir}/{pdb_code}/{pdb_code}_pocket.pdb

Output:
    ../data/pocket_pointclouds/{pdb_code}.npz
        One file per entry, containing:
        - positions: float32 (N, 3) — centroid-centered xyz in Angstroms
        - features:  float32 (N, 46) — per-atom chemical feature vector

    See documentation/pocket_pointcloud_features.md for full feature specification.

Feature vector (46 dimensions):
    [0-16]  Element type     — 17-class one-hot (C, N, O, S, P, H, ZN, MG, CA,
                                MN, FE, CO, CU, NI, K, NA, Other)
    [17-41] Residue category — 25-class one-hot (20 standard AAs, water, metal,
                                modified residue, cofactor, other)
    [42]    Is backbone      — binary
    [43]    Is H-bond donor  — binary (heavy atoms only)
    [44]    Is H-bond acceptor — binary
    [45]    Is aromatic      — binary

Preprocessing:
    - Discards crystallization artifacts (HG, YB, EU, IR, CS, SR, RB, IN, GA)
    - Discards selenomethionine residues (MSE)
    - Remaps cadmium → zinc (crystallographic substitution)
    - Treats deuterated water (DOD) as regular water (HOH)
    - Skips alternate conformations (keeps only altloc ' ' or 'A')
    - Centers coordinates at pocket centroid

Dependencies:
    pip install pandas numpy
"""

import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

# ---------------------------------------------------------------------------
# Paths (env-overridable so a non-PDBbind source, e.g. BioLiP, can be featurized
# with the identical logic: STRUCT_ROOT/{data_subdir}/{pdb_code}/{pdb_code}_pocket.pdb)
# ---------------------------------------------------------------------------
PDBBIND_DIR = Path(os.environ.get("STRUCT_ROOT", "data/structures"))
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
INPUT_CSV = Path(os.environ.get("PC_INPUT_CSV", str(DATA_DIR / "pdbbind_with_pockets.csv")))
OUTPUT_DIR = Path(os.environ.get("PC_OUTPUT_DIR", str(DATA_DIR / "pocket_pointclouds")))

# ---------------------------------------------------------------------------
# Constants: Element mapping
# ---------------------------------------------------------------------------

# Elements to discard entirely (crystallization artifacts)
DISCARD_ELEMENTS = {"HG", "YB", "EU", "IR", "CS", "SR", "RB", "IN", "GA"}

# Residues to discard entirely
DISCARD_RESIDUES = {"MSE"}

# Element remapping (crystallographic substitutions and isotopes)
ELEMENT_REMAP = {
    "CD": "ZN",  # Cadmium -> Zinc (crystallographic substitute)
    "D": "H",    # Deuterium -> Hydrogen (same chemistry, heavier isotope)
}

# Residue remapping
RESIDUE_REMAP = {"DOD": "HOH"}

# Element to one-hot index (17 categories)
# Non-metals first (C, N, O, S, P, H), then metals, then Other
ELEMENT_TO_INDEX = {
    "C": 0, "N": 1, "O": 2, "S": 3, "P": 4, "H": 5,
    "ZN": 6, "MG": 7, "CA": 8, "MN": 9, "FE": 10,
    "CO": 11, "CU": 12, "NI": 13, "K": 14, "NA": 15,
}
N_ELEMENT_CLASSES = 17  # index 16 = Other

# Metal elements (for residue categorization)
METAL_ELEMENTS = {
    "ZN", "MG", "CA", "MN", "FE", "CO", "CU", "NI", "K", "NA",
    "CD",  # before remapping, just in case
}

# ---------------------------------------------------------------------------
# Constants: Residue mapping
# ---------------------------------------------------------------------------

STANDARD_AMINO_ACIDS = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
AA_TO_INDEX = {aa: i for i, aa in enumerate(STANDARD_AMINO_ACIDS)}  # 0-19

# Residue category offsets in feature vector
RES_OFFSET = 17        # indices 17-41
RES_WATER = 20         # index 37 = 17 + 20
RES_METAL = 21         # index 38 = 17 + 21
RES_MODIFIED = 22      # index 39 = 17 + 22
RES_COFACTOR = 23      # index 40 = 17 + 23
RES_OTHER = 24         # index 41 = 17 + 24
N_RESIDUE_CLASSES = 25

# Modified amino acids (biological post-translational modifications)
MODIFIED_RESIDUES = {
    "KCX", "LLP", "CSO", "TPO", "SEP", "PTR", "CSD", "CME", "OCS", "ALY",
    "CSX", "CMH", "MLY", "CCS", "NLE", "DAL", "CAS", "PHD", "CSS", "PCA",
    "HYP", "TPQ", "M3L", "NEP", "AGM", "FME", "MHO", "SCH", "SMC", "SEC",
    "DVA", "AIB", "ABA", "ORN", "IAS", "DLE", "DAR", "MLE", "SAC", "GLZ",
    "CGU", "TYS", "CSE", "HIC", "BMT", "MEA", "5HP", "FTR", "FT6",
    "MIS", "MHS", "DNP", "DLY", "DHI", "DGN", "DGL", "DDG", "DCY",
    "CXM", "CMT", "CGA", "B3A", "ACY", "ACE", "7N8", "4H0", "4AK",
    "SNN", "SGB", "PHQ", "PPN", "LED", "KPI", "HEK", "GL3", "FES",
    "F43", "DTR", "DTH", "FUL", "NH2", "YCM", "TRQ", "YOF", "Y1",
    "SUN", "SNC", "SCY", "MTY", "MGN", "MGF",
}

# Water residues
WATER_RESIDUES = {"HOH", "DOD"}

# ---------------------------------------------------------------------------
# Constants: Binary features
# ---------------------------------------------------------------------------

# Backbone atom names (for standard and modified amino acids)
BACKBONE_ATOMS = {"N", "CA", "C", "O", "H", "HA"}

# H-bond donors: set of (resname, atom_name) tuples
# "backbone_N" is handled separately (all standard AAs except PRO)
SIDECHAIN_DONORS = {
    ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"),
    ("ASN", "ND2"),
    ("CYS", "SG"),
    ("GLN", "NE2"),
    ("HIS", "ND1"), ("HIS", "NE2"),
    ("LYS", "NZ"),
    ("SER", "OG"),
    ("THR", "OG1"),
    ("TRP", "NE1"),
    ("TYR", "OH"),
}
# Water oxygen is both donor and acceptor (handled in code)

# H-bond acceptors: all O atoms + specific N and S atoms
ACCEPTOR_N_ATOMS = {("HIS", "ND1"), ("HIS", "NE2")}
ACCEPTOR_S_ATOMS = {("MET", "SD"), ("CYS", "SG")}

# Aromatic atoms by residue
AROMATIC_ATOMS = {
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ",
            "HD1", "HD2", "HE1", "HE2", "HZ"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ",
            "HD1", "HD2", "HE1", "HE2"},
    "TRP": {"CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2",
            "HD1", "HE1", "HE3", "HZ2", "HZ3", "HH2"},
    "HIS": {"CG", "ND1", "CD2", "CE1", "NE2",
            "HD1", "HD2", "HE1", "HE2"},
}

FEATURE_DIM = 52   # 46 structural/chem one-hots + 6 physical channels (46-51)

# ---------------------------------------------------------------------------
# PDB parsing
# ---------------------------------------------------------------------------

def extract_element(line):
    """
    Extract the element symbol from a PDB ATOM/HETATM line.

    Primary source: columns 77-78 (official element field).
    Fallback: infer from atom name (columns 13-16) if element field is blank.
    """
    # Try element column first (cols 77-78, 0-indexed 76:78)
    if len(line) >= 78:
        elem = line[76:78].strip()
        # Remove charge indicators that sometimes bleed in (e.g., "2+")
        elem = "".join(c for c in elem if c.isalpha())
        if elem:
            return elem.upper()

    # Fallback: infer from atom name
    atom_name = line[12:16].strip()
    # Strip leading digits (older PDB format: "1HG1" -> "HG1" -> "H")
    name = atom_name.lstrip("0123456789")
    if not name:
        return "X"
    # Two-letter elements: check if first two chars are a known element
    if len(name) >= 2 and name[:2] in ELEMENT_TO_INDEX:
        return name[:2]
    if len(name) >= 2 and name[:2] in METAL_ELEMENTS:
        return name[:2]
    return name[0].upper()


def parse_pocket_pdb(pdb_path):
    """
    Parse a PDBbind pocket PDB file into positions and raw atom info.

    Returns:
        List of dicts, each with keys:
            element, resname, atom_name, x, y, z, is_hetatm
        Returns empty list if file not found or has no valid atoms.
    """
    atoms = []
    try:
        with open(pdb_path, "r") as f:
            for line in f:
                # Only ATOM and HETATM records
                is_hetatm = line.startswith("HETATM")
                if not line.startswith("ATOM") and not is_hetatm:
                    continue

                # Skip alternate conformations (keep only ' ' and 'A')
                altloc = line[16]
                if altloc not in (" ", "A", ""):
                    continue

                # Extract fields
                atom_name = line[12:16].strip()
                resname = line[17:20].strip()

                # Discard entire MSE residues
                if resname in DISCARD_RESIDUES:
                    continue

                # Remap residue names
                if resname in RESIDUE_REMAP:
                    resname = RESIDUE_REMAP[resname]

                # Extract and process element
                element = extract_element(line)
                element = ELEMENT_REMAP.get(element, element)

                # Discard crystallization artifact elements
                if element in DISCARD_ELEMENTS:
                    continue

                # Parse coordinates
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except (ValueError, IndexError):
                    continue

                atoms.append({
                    "element": element,
                    "resname": resname,
                    "atom_name": atom_name,
                    "x": x, "y": y, "z": z,
                    "is_hetatm": is_hetatm,
                })
    except FileNotFoundError:
        pass

    return atoms


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def get_element_index(element):
    """Map element symbol to one-hot index (0-15)."""
    if element in ELEMENT_TO_INDEX:
        return ELEMENT_TO_INDEX[element]
    # Handle FE2 -> FE, CU1 -> CU
    base = "".join(c for c in element if c.isalpha())
    if base in ELEMENT_TO_INDEX:
        return ELEMENT_TO_INDEX[base]
    return 16  # Other


def get_residue_index(resname, element, is_hetatm):
    """Map residue to one-hot index within the 25-class residue block (0-24)."""
    # Standard amino acid
    if resname in AA_TO_INDEX:
        return AA_TO_INDEX[resname]

    # Water
    if resname in WATER_RESIDUES:
        return RES_WATER

    # Metal ion (single-atom HETATM with metallic element)
    if is_hetatm and element in METAL_ELEMENTS:
        return RES_METAL

    # Modified amino acid
    if resname in MODIFIED_RESIDUES:
        return RES_MODIFIED

    # Cofactor (multi-atom HETATM not in above categories)
    if is_hetatm:
        return RES_COFACTOR

    # Fallback
    return RES_OTHER


def is_backbone(atom_name, resname):
    """Check if atom is a backbone atom of an amino acid."""
    if resname not in AA_TO_INDEX and resname not in MODIFIED_RESIDUES:
        return False
    return atom_name in BACKBONE_ATOMS


def is_hbond_donor(element, resname, atom_name):
    """Check if heavy atom is an H-bond donor."""
    # Water oxygen is a donor
    if resname in WATER_RESIDUES and element == "O":
        return True

    # Only check heavy atoms (not hydrogens)
    if element == "H":
        return False

    # Backbone amide N (all standard AAs except PRO)
    if atom_name == "N" and resname in AA_TO_INDEX and resname != "PRO":
        return True

    # Side chain donors
    if (resname, atom_name) in SIDECHAIN_DONORS:
        return True

    return False


def is_hbond_acceptor(element, resname, atom_name):
    """Check if atom is an H-bond acceptor."""
    # All oxygens are acceptors
    if element == "O":
        return True

    # Specific nitrogen acceptors
    if element == "N" and (resname, atom_name) in ACCEPTOR_N_ATOMS:
        return True

    # Specific sulfur acceptors
    if element == "S" and (resname, atom_name) in ACCEPTOR_S_ATOMS:
        return True

    return False


def is_aromatic(resname, atom_name):
    """Check if atom is part of an aromatic ring."""
    if resname in AROMATIC_ATOMS:
        return atom_name in AROMATIC_ATOMS[resname]
    return False


# ---------------------------------------------------------------------------
# Physical property channels (indices 46-51) — added to make the pocket's
# electrostatics / hydrophobicity / shape explicit for the model (ESM + one-hots
# encode identity, not physics). Each is a robust structure-only quantity.
# ---------------------------------------------------------------------------

# Kyte-Doolittle hydropathy (per residue)
KD_HYDRO = {"ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, "CYS": 2.5, "MET": 1.9,
            "ALA": 1.8, "GLY": -0.4, "THR": -0.7, "SER": -0.8, "TRP": -0.9,
            "TYR": -1.3, "PRO": -1.6, "HIS": -3.2, "GLU": -3.5, "GLN": -3.5,
            "ASP": -3.5, "ASN": -3.5, "LYS": -3.9, "ARG": -4.5}
# Eisenberg consensus hydrophobicity (per residue)
EIS_HYDRO = {"ILE": 1.38, "PHE": 1.19, "VAL": 1.08, "LEU": 1.06, "TRP": 0.81,
             "MET": 0.64, "ALA": 0.62, "CYS": 0.29, "GLY": 0.48, "TYR": 0.26,
             "THR": -0.05, "SER": -0.18, "HIS": -0.40, "GLN": -0.85, "LYS": -1.50,
             "ASN": -0.78, "GLU": -0.74, "PRO": 0.12, "ASP": -0.90, "ARG": -2.53}


def partial_charge(resname, aname, element):
    """Coarse but chemically-grounded partial charge (dominant electrostatics)."""
    if resname == "ASP" and aname in ("OD1", "OD2"): return -0.5
    if resname == "GLU" and aname in ("OE1", "OE2"): return -0.5
    if resname == "ASP" and aname == "CG": return 0.5
    if resname == "GLU" and aname == "CD": return 0.5
    if resname == "LYS" and aname == "NZ": return 1.0
    if resname == "ARG" and aname in ("NH1", "NH2", "NE"): return 0.33
    if resname == "ARG" and aname == "CZ": return 0.4
    if resname == "HIS" and aname in ("ND1", "NE2"): return 0.25
    if resname == "SER" and aname == "OG": return -0.4
    if resname == "THR" and aname == "OG1": return -0.4
    if resname == "TYR" and aname == "OH": return -0.4
    if resname == "CYS" and aname == "SG": return -0.2
    if resname == "TRP" and aname == "NE1": return -0.3
    if resname == "ASN" and aname == "OD1": return -0.5
    if resname == "ASN" and aname == "ND2": return -0.3
    if resname == "GLN" and aname == "OE1": return -0.5
    if resname == "GLN" and aname == "NE2": return -0.3
    if aname == "O": return -0.5          # backbone carbonyl O
    if aname == "N": return -0.3          # backbone amide N
    if aname == "C": return 0.5           # backbone carbonyl C
    if aname == "CA": return 0.1
    if element in METAL_ELEMENTS: return 2.0
    return 0.0


def atoms_to_arrays(atoms):
    """
    Convert list of atom dicts to positions and features arrays.

    Returns:
        positions: float32 (N, 3) — centroid-centered coordinates
        features:  float32 (N, 52) — per-atom feature vector
                   (indices 0-45 structural/chem one-hots; 46-51 physical)
    """
    n = len(atoms)
    positions = np.zeros((n, 3), dtype=np.float32)
    features = np.zeros((n, FEATURE_DIM), dtype=np.float32)

    for i, atom in enumerate(atoms):
        elem = atom["element"]
        resname = atom["resname"]
        aname = atom["atom_name"]
        is_het = atom["is_hetatm"]

        # Positions
        positions[i] = [atom["x"], atom["y"], atom["z"]]

        # Element one-hot (indices 0-15)
        features[i, get_element_index(elem)] = 1.0

        # Residue one-hot (indices 16-40)
        features[i, RES_OFFSET + get_residue_index(resname, elem, is_het)] = 1.0

        # Is backbone (index 42)
        if is_backbone(aname, resname):
            features[i, 42] = 1.0

        # Is H-bond donor (index 43)
        if is_hbond_donor(elem, resname, aname):
            features[i, 43] = 1.0

        # Is H-bond acceptor (index 44)
        if is_hbond_acceptor(elem, resname, aname):
            features[i, 44] = 1.0

        # Is aromatic (index 45)
        if is_aromatic(resname, aname):
            features[i, 45] = 1.0

    # Center at centroid
    centroid = positions.mean(axis=0)
    positions -= centroid

    # --- Physical channels (46-51): electrostatics, hydrophobicity, shape ---
    from scipy.spatial.distance import cdist
    from scipy.spatial import cKDTree
    charges = np.array([partial_charge(a["resname"], a["atom_name"], a["element"])
                        for a in atoms], dtype=np.float32)
    features[:, 46] = charges                                    # [46] partial charge
    if n > 1:
        r = cdist(positions, positions).astype(np.float32)
        np.fill_diagonal(r, np.inf)
        ep = (charges[None, :] / (r + 1.0)).sum(axis=1)          # [47] Coulombic potential
        features[:, 47] = np.tanh(ep / 5.0)
        tree = cKDTree(positions)                                # [50] burial (local density)
        counts = tree.query_ball_point(positions, 8.0)
        features[:, 50] = np.tanh(
            np.array([len(c) - 1 for c in counts], dtype=np.float32) / 40.0)
        metal_mask = np.array([a["element"] in METAL_ELEMENTS for a in atoms])
        if metal_mask.any():                                     # [51] near-metal flag
            dmin, _ = cKDTree(positions[metal_mask]).query(positions, k=1)
            features[:, 51] = (dmin <= 3.5).astype(np.float32)
    features[:, 48] = np.array([KD_HYDRO.get(a["resname"], 0.0) / 4.5    # [48] KD hydrophobicity
                                for a in atoms], dtype=np.float32)
    features[:, 49] = np.array([EIS_HYDRO.get(a["resname"], 0.0) / 2.5   # [49] Eisenberg
                                for a in atoms], dtype=np.float32)

    return positions, features


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--relaxed",
        action="store_true",
        help="Read pdbbind_with_pockets_relaxed.csv. Point cloud .npz files "
             "are written to the same shared dir (keyed by pdb_code), so "
             "existing files from the original run are reused unmodified.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip pdb codes whose .npz already exists (delta processing).",
    )
    args = parser.parse_args()

    input_csv = INPUT_CSV if os.environ.get("PC_INPUT_CSV") else DATA_DIR / (
        "pdbbind_with_pockets_relaxed.csv" if args.relaxed
        else "pdbbind_with_pockets.csv"
    )
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} entries from {input_csv.name}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Tracking
    failed = []
    empty = []
    atom_counts = []
    skipped = 0
    element_counter = Counter()
    residue_counter = Counter()
    n_with_waters = 0
    n_with_metals = 0
    n_with_modified = 0
    n_with_cofactors = 0
    discarded_element_counter = Counter()
    discarded_mse_count = 0

    for i, row in df.iterrows():
        pdb_code = row["pdb_code"]
        data_subdir = row["data_subdir"]

        out_path = OUTPUT_DIR / f"{pdb_code}.npz"
        if args.skip_existing and out_path.exists():
            skipped += 1
            continue

        pdb_path = PDBBIND_DIR / data_subdir / pdb_code / f"{pdb_code}_pocket.pdb"
        atoms = parse_pocket_pdb(pdb_path)

        if not atoms:
            if not pdb_path.exists():
                failed.append(pdb_code)
            else:
                empty.append(pdb_code)
            continue

        positions, features = atoms_to_arrays(atoms)
        atom_counts.append(len(atoms))

        # Track statistics
        for atom in atoms:
            element_counter[atom["element"]] += 1
            residue_counter[atom["resname"]] += 1

        has_water = any(a["resname"] in WATER_RESIDUES for a in atoms)
        has_metal = any(a["element"] in METAL_ELEMENTS for a in atoms)
        has_modified = any(a["resname"] in MODIFIED_RESIDUES for a in atoms)
        has_cofactor = any(
            a["is_hetatm"]
            and a["resname"] not in WATER_RESIDUES
            and a["element"] not in METAL_ELEMENTS
            and a["resname"] not in MODIFIED_RESIDUES
            for a in atoms
        )
        if has_water:
            n_with_waters += 1
        if has_metal:
            n_with_metals += 1
        if has_modified:
            n_with_modified += 1
        if has_cofactor:
            n_with_cofactors += 1

        # Save (out_path already computed above for skip-existing check)
        np.savez_compressed(out_path, positions=positions, features=features)

        # Progress
        if (i + 1) % 2000 == 0:
            print(f"  Processed {i + 1}/{len(df)} entries "
                  f"({len(failed)} failures so far)")

    # --- Summary ---
    n_success = len(atom_counts)
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Entries processed: {n_success}/{len(df)}")
    print(f"Skipped (existing):{skipped}")
    print(f"File not found:    {len(failed)}")
    print(f"Empty after filter:{len(empty)}")

    if atom_counts:
        ac = np.array(atom_counts)
        print(f"\nAtom counts per pocket:")
        print(f"  Mean:   {ac.mean():.0f}")
        print(f"  Median: {np.median(ac):.0f}")
        print(f"  Min:    {ac.min()}")
        print(f"  Max:    {ac.max()}")
        print(f"  Std:    {ac.std():.0f}")

    if n_success > 0:   # guard: all-skipped restart leaves n_success=0
        print(f"\nEntries containing:")
        print(f"  Waters:            {n_with_waters} ({100*n_with_waters/n_success:.1f}%)")
        print(f"  Metal ions:        {n_with_metals} ({100*n_with_metals/n_success:.1f}%)")
        print(f"  Modified residues: {n_with_modified} ({100*n_with_modified/n_success:.1f}%)")
        print(f"  Cofactors:         {n_with_cofactors} ({100*n_with_cofactors/n_success:.1f}%)")

    print(f"\nElement distribution (top 20):")
    for elem, count in element_counter.most_common(20):
        print(f"  {elem:>4s}: {count:>10,}")

    print(f"\nResidue distribution (top 30):")
    for res, count in residue_counter.most_common(30):
        print(f"  {res:>5s}: {count:>10,}")

    if failed:
        print(f"\nFailed PDB codes (file not found): {failed[:20]}")

    if empty:
        print(f"\nEmpty pockets (all atoms filtered): {empty[:20]}")

    print(f"\nSaved {n_success} point clouds to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
