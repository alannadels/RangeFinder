"""
BioLiP integration -- Stage 2 (structure bridge). Runs on the instance (needs
network + openbabel).

For each pilot entry (pdb, rec_chain, ccd, lig_chain, lig_serial, lig_resnum):
  1. download the RCSB structure (files.rcsb.org, cached),
  2. locate the ligand (HETATM resName==ccd, chain==lig_chain),
  3. cut the pocket = every protein/water/ion residue with an atom within RADIUS
     of any ligand atom (whole residues; ligand itself excluded),
  4. protonate with openbabel,
  5. write data/biolip/structures/{id}_pocket.pdb  (PDBbind-format pocket file),

so the existing compute_pocket_pointclouds / compute_esm scripts run on it.
Also writes data/biolip/biolip_pockets.csv  (pdb_code=id, data_subdir, uniprot,
pocket_id) -- the CSV the downstream stages consume.

    python -m scripts.build_biolip_structures [--limit N]
"""
import argparse, os, subprocess, urllib.request
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BL = ROOT / "data" / "biolip"
# raw PDB cache is regenerable scratch -> put on free local NVMe if BIOLIP_SCRATCH set
PDBDIR = Path(os.environ.get("BIOLIP_SCRATCH", str(BL / "pdb"))); PDBDIR.mkdir(parents=True, exist_ok=True)
STRUCT = BL / "structures"; STRUCT.mkdir(parents=True, exist_ok=True)
RADIUS = 10.0
MAX_HEAVY = 1200   # cap heavy atoms (closest residues to ligand); ~2400 post-protonation,
                   # matching PDBbind's size distribution so the SE(3) edge graph stays bounded
MAX_PDB_BYTES = 12_000_000   # skip giant assemblies (ribosomes/capsids ~300k atoms) that
                             # OOM the box when parsed in parallel; edge cases we don't want
AA3 = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS",
       "MET","PHE","PRO","SER","THR","TRP","TYR","VAL","MSE","SEC","PYL"}
KEEP_HET = {"HOH","DOD","ZN","MG","CA","MN","FE","FE2","CU","CU1","NA","K","NI","CO"}


def download_pdb(pdb):
    fp = PDBDIR / f"{pdb}.pdb"
    if fp.exists() and fp.stat().st_size > 0:
        return fp
    try:
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb}.pdb", fp)
        return fp if fp.stat().st_size > 0 else None
    except Exception:
        return None


def parse_atoms(fp):
    """Return list of dicts for ATOM/HETATM records (first altloc only)."""
    out = []
    for ln in fp.read_text(errors="ignore").splitlines():
        rec = ln[:6].strip()
        if rec not in ("ATOM", "HETATM"):
            continue
        alt = ln[16]
        if alt not in (" ", "A"):
            continue
        try:
            xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
        except ValueError:
            continue
        out.append({"rec": rec, "name": ln[12:16].strip(), "res": ln[17:20].strip(),
                    "chain": ln[21], "resseq": ln[22:26].strip(), "icode": ln[26],
                    "xyz": xyz, "line": ln})
    return out


def cut_pocket(atoms, ccd, lig_chain, lig_resnum):
    # Isolate the SPECIFIC ligand copy BioLiP annotated (a CCD can appear many times
    # in one structure -> matching by resName+chain alone merges all copies into one
    # bogus, oversized pocket). Match chain + residue number, with robust fallbacks.
    rs = str(lig_resnum).strip()
    het = [a for a in atoms if a["rec"] == "HETATM" and a["res"] == ccd]
    lig = [a for a in het if a["chain"] == lig_chain and a["resseq"].strip() == rs]
    if not lig:                                   # fallback: chain match, first copy
        ch = [a for a in het if a["chain"] == lig_chain]
        if ch:
            rs0 = ch[0]["resseq"]; lig = [a for a in ch if a["resseq"] == rs0]
    if not lig and het:                           # fallback: any chain, first copy
        k0 = (het[0]["chain"], het[0]["resseq"]); lig = [a for a in het if (a["chain"], a["resseq"]) == k0]
    if not lig:
        return None, 0
    L = np.array([a["xyz"] for a in lig])
    lig_key = {(a["chain"], a["resseq"], a["icode"], a["res"]) for a in lig}

    # candidate receptor atoms (protein + kept het), excluding the ligand residues
    cand = [a for a in atoms
            if (a["res"] in AA3 or a["res"] in KEEP_HET)
            and (a["chain"], a["resseq"], a["icode"], a["res"]) not in lig_key]
    if not cand:
        return None, 0
    C = np.array([a["xyz"] for a in cand])
    dmin2 = ((C[:, None, :] - L[None, :, :]) ** 2).sum(-1).min(1)   # per-atom min dist^2 to ligand
    # group ALL candidate atoms by residue; residue distance = its closest atom
    res_all, res_d = {}, {}
    for i, a in enumerate(cand):
        k = (a["chain"], a["resseq"], a["icode"], a["res"])
        res_all.setdefault(k, []).append(i)
        res_d[k] = min(res_d.get(k, 1e18), dmin2[i])
    # residues with any atom within RADIUS, ordered nearest-first
    near_res = sorted([k for k in res_all if res_d[k] <= RADIUS ** 2], key=res_d.get)
    # accumulate WHOLE residues closest to the ligand until the heavy-atom cap
    lines, kept = [], 0
    for k in near_res:
        idxs = res_all[k]
        if kept and kept + len(idxs) > MAX_HEAVY:
            break
        lines += [cand[i]["line"] for i in idxs]; kept += len(idxs)
    return lines, kept


def process_entry(r):
    """Download -> cut specific-copy pocket -> protonate -> pocket.pdb. Returns row or None."""
    eid = f"{r.pdb}_{r.rec_chain}_{r.ccd}_{r.lig_serial}"
    edir = STRUCT / eid; edir.mkdir(exist_ok=True)
    pocket_H = edir / f"{eid}_pocket.pdb"
    if not pocket_H.exists():
        fp = download_pdb(r.pdb)
        if fp is None or fp.stat().st_size > MAX_PDB_BYTES:
            return None                       # skip missing / giant-assembly structures
        lines, nres = cut_pocket(parse_atoms(fp), r.ccd, r.lig_chain, r.lig_resnum)
        if not lines or nres < 5:
            return None
        raw = edir / f"{eid}_raw.pdb"
        raw.write_text("\n".join(lines) + "\nEND\n")
        # protonate with openbabel (pH 7.4); fall back to heavy-atom pocket
        try:
            subprocess.run(["obabel", str(raw), "-O", str(pocket_H), "-h", "-p", "7.4"],
                           check=True, capture_output=True, timeout=120)
        except Exception:
            pocket_H.write_text(raw.read_text())
        raw.unlink(missing_ok=True)
    return {"pdb_code": eid, "data_subdir": "structures",
            "uniprot_id": r.uni, "pocket_id": f"{r.uni}_1"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    entries_csv = Path(os.environ.get("ENTRIES_CSV", str(BL / "pilot_entries.csv")))
    ent = pd.read_csv(entries_csv, dtype=str)
    print(f"entries: {entries_csv} ({len(ent)} rows)")
    if args.limit:
        ent = ent.head(args.limit)
    recs = [r for _, r in ent.iterrows()]

    from concurrent.futures import ThreadPoolExecutor
    # Phase 1: parallel-prefetch unique PDBs (I/O-bound -> big speedup at full scale)
    pdbs = list(ent.pdb.unique())
    print(f"prefetching {len(pdbs)} unique PDBs with {args.workers} workers...")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, _ in enumerate(ex.map(download_pdb, pdbs)):
            if (i + 1) % 2000 == 0:
                print(f"  downloaded {i+1}/{len(pdbs)}")

    # Phase 2: cut + protonate. Capped at 4 workers -- each parse holds a whole
    # structure in RAM, and many large ones in parallel OOM the 15 GB box.
    rows, ok, fail = [], 0, 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, res in enumerate(ex.map(process_entry, recs)):
            if res:
                rows.append(res); ok += 1
            else:
                fail += 1
            if (i + 1) % 500 == 0:
                print(f"  processed {i+1}/{len(recs)}  ok={ok} fail={fail}")
    pockets_out = Path(os.environ.get("POCKETS_OUT_CSV", str(BL / "biolip_pockets.csv")))
    pd.DataFrame(rows).to_csv(pockets_out, index=False)
    print(f"DONE: {ok} pockets featurizable, {fail} failed -> {pockets_out}")


if __name__ == "__main__":
    main()
