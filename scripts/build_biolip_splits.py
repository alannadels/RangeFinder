"""
BioLiP integration -- Stage 3: protein-grouped train/val/test splits.
Splits by UniProt (no leakage), 70/15/15, seed 42. Only pockets that were
successfully featurized (have a .npz) are included.

    python -m scripts.build_biolip_splits
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BL = ROOT / "data" / "biolip"
PCDIR = BL / "pointclouds_esm"
SEED = 42


def main():
    ent = pd.read_csv(BL / "biolip_pockets.csv", dtype=str)
    dist = pd.read_csv(BL / "pilot_pocket_distributions.csv")
    # keep only entries whose featurized .npz exists AND isn't an oversized pocket
    # (belt-and-suspenders; the structure bridge now caps size, but this drops any
    # already-featurized monster that would OOM the SE(3) edge graph)
    # GPU memory is driven by EDGES (atom pairs within the 5A SE(3) graph), not atoms:
    # dense pockets have 10-30x more edges/atom. Drop the ~0.4% pathologically dense
    # pockets that OOM even alone (e.g. 720k edges vs ~23k typical), and record n_edges
    # for edge-budget batching.
    from scipy.spatial import cKDTree
    R, MAX_ATOMS, EDGE_CAP = 5.0, 2000, 50000
    natoms, nedges = {}, {}
    for p in glob.glob(str(PCDIR / "*.npz")):
        try:
            pos = np.load(p)["positions"]; n = pos.shape[0]
            if n > MAX_ATOMS:
                continue
            e = int(len(cKDTree(pos).query_pairs(R)))
            if e > EDGE_CAP:
                continue                                   # too dense to fit on the GPU alone
            natoms[Path(p).stem] = int(n); nedges[Path(p).stem] = e
        except Exception:
            pass
    have = set(natoms)
    ent = ent[ent.pdb_code.isin(have)].copy()
    ent["n_atoms"] = ent.pdb_code.map(natoms)
    ent["n_edges"] = ent.pdb_code.map(nedges)              # edge count = memory driver, used for batching
    ent = ent.merge(dist[["pocket_id", "n_ligands"]], on="pocket_id", how="left")
    print(f"featurized entries: {len(ent)} | pockets: {ent.pocket_id.nunique()} "
          f"| proteins: {ent.uniprot_id.nunique()}")

    # Only pockets with >=3 ligands have a real distribution (non-zero sigma) and are
    # eligible for train/val (GIoU needs sigma). Pockets with 1-2 ligands go to TEST
    # only -- usable for coverage eval, never trained on -- matching create_splits.py.
    rng = np.random.RandomState(SEED)
    elig = sorted(ent[ent.n_ligands >= 3].uniprot_id.unique())
    small = sorted(ent[ent.n_ligands < 3].uniprot_id.unique())
    rng.shuffle(elig)
    n = len(elig); ntr = int(0.70 * n); nva = int(0.15 * n)
    grp = {u: ("train" if i < ntr else "val" if i < ntr + nva else "test")
           for i, u in enumerate(elig)}
    for u in small:
        grp[u] = "test"                       # 1-2 lig pockets: test-only (no leakage)
    ent["split"] = ent.uniprot_id.map(grp)
    print(f"eligible (>=3 lig) proteins: {len(elig)} | small (1-2 lig, test-only): {len(small)}")
    for s in ["train", "val", "test"]:
        sub = ent[ent.split == s]
        sub.to_csv(BL / f"{s}.csv", index=False)
        n3 = sub[sub.n_ligands >= 3].pocket_id.nunique()
        print(f"  {s}: {len(sub)} entries / {sub.pocket_id.nunique()} pockets "
              f"({n3} with >=3 lig)")


if __name__ == "__main__":
    main()
