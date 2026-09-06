"""
BioLiP integration -- Stage 1 (ligands, no structures/GPU needed).

From the BioLiP annotation index:
  1. filter to regular small-molecule ligands (drop bio-polymers, metals, ions,
     buffers, and common cofactors/sugars),
  2. select a PILOT set of proteins (pocket == UniProt for the pilot),
  3. resolve each ligand CCD code -> SMILES via the RCSB GraphQL batch API,
  4. compute the SAME 9 RDKit descriptors as compute_ligand_properties.py,
  5. apply a light drug-like MW filter,
  6. aggregate per-protein (pilot pocket) mean/std -> pilot pocket_distributions.

Outputs (data/biolip/):
  ccd_props.csv          ccd, smiles, + 9 properties
  pilot_entries.csv      pdb, rec_chain, ccd, lig_chain, lig_serial, uniprot  (structures to featurize)
  pilot_pocket_distributions.csv   pocket_id(=uniprot), n_ligands, {prop}_mean/std/...

    python -m scripts.build_biolip_ligands [--full] [--n-proteins 400]
"""
import argparse, gzip, json, sys, time, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"; BL = DATA / "biolip"; BL.mkdir(exist_ok=True)
PROPS = ["MW", "logP", "HBD", "HBA", "TPSA", "rotatable_bonds",
         "formal_charge", "aromatic_rings", "fsp3"]
GRAPHQL = "https://data.rcsb.org/graphql"

# ligand categories to EXCLUDE (not drug-like binders)
NONSM = {"peptide", "dna", "rna"}
METALS = {"ZN","MG","CA","NA","K","FE","FE2","MN","CU","CU1","CO","NI","CD","HG","BA","SR","CS","RB","LI","AL","AU","AG","PT","PB","PD","MO","W","V","CR","TL","GD","EU","YB","SM","HO","ND","LA","CE","PR","TB","DY","ER","TM","LU","Y","SC","ZR","NB","TC","RU","RH","OS","IR","RE","HF","TA","GA","IN","SN","SB","TE","BI","BE"}
IONS_BUF = {"CL","BR","IOD","FLO","SO4","PO4","NO3","ACT","EDO","GOL","PEG","PG4","PGE","DMS","MES","TRS","EPE","FLC","CIT","BME","MPD","IPA","NH4","OH","O","OXY","CO3","FMT","ACY","TAR","MLI","SCN","AZI","NCO","DTT","BOG","P6G","1PE","2PE","TES","POP"}
# common cofactors/sugars -- biologically real but not drug-like ligands
COFAC_SUGAR = {"HEM","HEC","HEA","SF4","FES","FAD","FMN","NAD","NAP","NDP","NAI","ATP","ADP","AMP","ANP","GTP","GDP","GNP","GSP","COA","ACO","SAM","SAH","PLP","TPP","BTN","MTE","MGD","B12","CLA","BCL","CHL","UDP","UMP","UPG","U5P",
                "NAG","BGC","GLC","MAN","GAL","BMA","FUC","XYS","NGA","SIA","GLA","FUL","BOG","LMT","SUC","MAL","TRE","GLO"}
EXCLUDE = METALS | IONS_BUF | COFAC_SUGAR


def load_biolip():
    cols = {0: "pdb", 1: "rec_chain", 4: "ccd", 5: "lig_chain", 6: "lig_serial",
            19: "lig_resnum", 17: "uni"}
    bl = pd.read_csv(BL / "BioLiP.txt.gz", sep="\t", header=None,
                     usecols=list(cols), names=[cols[i] for i in sorted(cols)],
                     dtype=str, keep_default_na=False)
    bl = bl[~bl.ccd.isin(NONSM | EXCLUDE)]           # regular small-molecule, drug-like-ish
    bl = bl[bl.uni != ""]
    return bl


def fetch_smiles(ccds, batch=400):
    """RCSB GraphQL batch: CCD -> SMILES_stereo (fallback SMILES)."""
    out = {}
    q = ("query($ids:[String!]!){chem_comps(comp_ids:$ids){rcsb_id "
         "rcsb_chem_comp_descriptor{SMILES_stereo SMILES}}}")
    ccds = list(ccds)
    for i in range(0, len(ccds), batch):
        chunk = ccds[i:i + batch]
        body = json.dumps({"query": q, "variables": {"ids": chunk}}).encode()
        req = urllib.request.Request(GRAPHQL, data=body,
                                     headers={"Content-Type": "application/json"})
        for attempt in range(4):
            try:
                r = json.loads(urllib.request.urlopen(req, timeout=60).read())
                break
            except Exception as e:
                if attempt == 3:
                    print(f"  batch {i} failed: {e}"); r = {"data": {"chem_comps": []}}
                else:
                    time.sleep(2 * (attempt + 1))
        for c in (r.get("data", {}).get("chem_comps") or []):
            d = c.get("rcsb_chem_comp_descriptor") or {}
            smi = d.get("SMILES_stereo") or d.get("SMILES")
            if smi:
                out[c["rcsb_id"]] = smi
        print(f"  SMILES {min(i+batch,len(ccds))}/{len(ccds)} (resolved {len(out)})")
    return out


def props_from_smiles(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return {"MW": Descriptors.MolWt(m), "logP": Crippen.MolLogP(m),
            "HBD": Descriptors.NumHDonors(m), "HBA": Descriptors.NumHAcceptors(m),
            "TPSA": Descriptors.TPSA(m), "rotatable_bonds": Descriptors.NumRotatableBonds(m),
            "formal_charge": Chem.GetFormalCharge(m), "aromatic_rings": Descriptors.NumAromaticRings(m),
            "fsp3": Descriptors.FractionCSP3(m)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="all proteins (else pilot)")
    ap.add_argument("--n-proteins", type=int, default=400)
    ap.add_argument("--max-struct-per-protein", type=int, default=5)
    args = ap.parse_args()

    bl = load_biolip()
    print(f"BioLiP regular small-molecule interactions (pre-druglike): {len(bl)}")

    # per-protein ligand-diversity, to pick informative pilot proteins
    per = bl.groupby("uni").ccd.nunique().sort_values(ascending=False)
    ours = set(pd.read_csv(DATA / "pocket_distributions.csv")["uniprot_id"].astype(str))
    if args.full:
        keep_uni = set(per[per >= 3].index)                       # >=3 distinct ligands
    else:
        # pilot: proteins with 5-60 distinct ligands (real range, not mega-target);
        # bias to include overlap with our current set so we can show enrichment
        cand = per[(per >= 5) & (per <= 60)]
        overlap = [u for u in cand.index if u in ours]
        newp = [u for u in cand.index if u not in ours]
        half = args.n_proteins // 2
        keep_uni = set(overlap[:half] + newp[:args.n_proteins - min(half, len(overlap))])
    bl = bl[bl.uni.isin(keep_uni)]
    print(f"selected {len(keep_uni)} proteins (pilot={not args.full}) -> {len(bl)} interactions")

    # ---- CCD -> SMILES -> props ----
    ccds = sorted(bl.ccd.unique())
    print(f"resolving {len(ccds)} unique CCD codes -> SMILES ...")
    smi = fetch_smiles(ccds)
    rows = []
    for c in ccds:
        s = smi.get(c)
        p = props_from_smiles(s) if s else None
        if p:
            rows.append({"ccd": c, "smiles": s, **p})
    cprops = pd.DataFrame(rows)
    # drug-like MW window
    cprops = cprops[(cprops.MW >= 150) & (cprops.MW <= 900)]
    cprops.to_csv(BL / "ccd_props.csv", index=False)
    print(f"ccd_props.csv: {len(cprops)} drug-like ligands with computed properties")

    # keep only interactions whose ligand survived
    good = set(cprops.ccd)
    bl = bl[bl.ccd.isin(good)]

    # ---- pilot pocket == uniprot; entries = capped structures per protein ----
    bl = bl.merge(cprops, on="ccd", how="left")
    # entry list to featurize (cap structures per protein for the pilot)
    ent = bl.drop_duplicates(["pdb", "rec_chain", "ccd", "lig_chain", "lig_serial"])
    ent = ent.groupby("uni", group_keys=False).apply(
        lambda g: g.drop_duplicates("pdb").head(args.max_struct_per_protein))
    ent[["pdb", "rec_chain", "ccd", "lig_chain", "lig_serial", "lig_resnum", "uni"]].to_csv(
        BL / "pilot_entries.csv", index=False)
    print(f"pilot_entries.csv: {len(ent)} structures to featurize "
          f"({ent.uni.nunique()} pockets)")

    # ---- pocket distributions from ALL (deduped) ligands of each protein ----
    lig = bl.drop_duplicates(["uni", "ccd"])          # unique ligand per pocket
    recs = []
    for uni, g in lig.groupby("uni"):
        r = {"pocket_id": f"{uni}_1", "uniprot_id": uni, "n_ligands": len(g)}
        for p in PROPS:
            v = g[p].astype(float)
            r[f"{p}_mean"] = v.mean(); r[f"{p}_std"] = v.std(ddof=1) if len(v) > 1 else 0.0
            r[f"{p}_median"] = v.median(); r[f"{p}_min"] = v.min(); r[f"{p}_max"] = v.max()
            r[f"{p}_count"] = len(v)
        recs.append(r)
    dist = pd.DataFrame(recs)
    dist.to_csv(BL / "pilot_pocket_distributions.csv", index=False)
    print(f"pilot_pocket_distributions.csv: {len(dist)} pockets")
    print(f"  n_ligands: median={dist.n_ligands.median():.0f} "
          f">=3={(dist.n_ligands>=3).sum()} >=10={(dist.n_ligands>=10).sum()}")


if __name__ == "__main__":
    main()
