"""
Evaluate the trained BioLiP model: per-pocket range IoU on the >=3-lig test
pockets (the same 564 the cluster baselines used), so it's directly comparable to
  global-avg floor 0.402 | honest pooled-cluster 0.403 | oracle 0.466.

    python -m scripts.eval_biolip
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from model.config import Config
from model.dataset import PocketDataset, create_dataloader
from model.architecture import PocketSE3Transformer
from model.utils import load_checkpoint
from model.evaluate import run_inference

BL = PROJECT_ROOT / "data" / "biolip"
PROPS = ["MW", "logP", "HBD", "HBA", "TPSA", "rotatable_bonds",
         "formal_charge", "aromatic_rings", "fsp3"]


def iou_range(mu_a, sd_a, mu_b, sd_b):
    la, ha = mu_a - 2*sd_a, mu_a + 2*sd_a
    lb, hb = mu_b - 2*sd_b, mu_b + 2*sd_b
    inter = np.maximum(0, np.minimum(ha, hb) - np.maximum(la, lb))
    uni = (ha-la) + (hb-lb) - inter
    return np.where(uni > 0, inter/np.where(uni > 0, uni, 1), 0.0).mean(1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # optional: --checkpoint-dir <name-or-path> to eval a different run (e.g. the
    # denoising-pretrained fine-tune) against the same 564-pocket test set.
    ckdir = PROJECT_ROOT / "model" / "checkpoints" / "biolip_pilot"
    if "--checkpoint-dir" in sys.argv:
        val = sys.argv[sys.argv.index("--checkpoint-dir") + 1]
        p = Path(val)
        ckdir = p if p.is_absolute() else PROJECT_ROOT / "model" / "checkpoints" / val
    ck_path = sorted(ckdir.glob("se3_pocket_ligand_best_*.pt"))[-1]
    # optional: --pointcloud-dir <name> (e.g. pointclouds_esm52 for the physical-feature run)
    pc_dir = BL / "pointclouds_esm"
    if "--pointcloud-dir" in sys.argv:
        pc_dir = BL / sys.argv[sys.argv.index("--pointcloud-dir") + 1]
    # optional: --distributions <name> (e.g. pilot_pocket_distributions_affwt.csv)
    dist_csv = BL / "pilot_pocket_distributions.csv"
    if "--distributions" in sys.argv:
        dist_csv = BL / sys.argv[sys.argv.index("--distributions") + 1]
    # auto-infer the per-atom input width from the checkpoint (1326 vs 1332) so the model
    # is built to match, then subtract ESM (1280) to get the config base that __post_init__
    # re-expands.
    _raw = torch.load(ck_path, map_location="cpu", weights_only=False)
    in_full = _raw["model_state_dict"]["input_embed.linear.weight"].shape[1]
    cfg = Config(use_iou_loss=True, checkpoint_dir=ckdir,
                 input_features=in_full - 1280,
                 test_csv=BL / "test.csv",
                 pocket_distributions_csv=dist_csv,
                 pointcloud_dir=pc_dir,
                 atom_budget=80000)                     # edge-budget loader (avoid OOM)
    print(f"input width: {in_full} (base {in_full-1280} + ESM 1280) | pointclouds: {pc_dir.name} | dist: {dist_csv.name}")
    model = PocketSE3Transformer(cfg).to(device)
    ck = load_checkpoint(ck_path, model)
    tm, ts = np.asarray(ck["target_mean"]), np.asarray(ck["target_std"])
    print(f"checkpoint: {ck_path.name}  val_loss={ck.get('val_loss', float('nan')):.4f}")

    ds = PocketDataset(csv_path=cfg.test_csv, distributions_csv_path=cfg.pocket_distributions_csv,
                       pointcloud_dir=cfg.pointcloud_dir, target_columns=cfg.target_columns,
                       target_mean=tm, target_std=ts, sample_weights=None)
    loader = create_dataloader(ds, cfg.batch_size, shuffle=False,
                               num_workers=cfg.num_workers, atom_budget=cfg.atom_budget)
    out = run_inference(model, loader, device, collect_geom=False)
    preds = np.asarray(out[0]) * ts + tm
    df = pd.DataFrame(preds, columns=[f"pred_{c}" for c in cfg.target_columns])
    df["pocket_id"] = out[6]

    pmu = df.groupby("pocket_id")[[f"pred_{p}_mean" for p in PROPS]].mean()
    psd = df.groupby("pocket_id")[[f"pred_{p}_std" for p in PROPS]].mean().abs()
    dist = pd.read_csv(cfg.pocket_distributions_csv).set_index("pocket_id")
    ids = pmu.index
    nlig = dist.loc[ids, "n_ligands"].to_numpy()
    gmu = dist.loc[ids, [f"{p}_mean" for p in PROPS]].to_numpy()
    gsd = dist.loc[ids, [f"{p}_std" for p in PROPS]].to_numpy()
    P, S = pmu.to_numpy(), psd.to_numpy()

    m3 = nlig >= 3
    io3 = iou_range(P[m3], S[m3], gmu[m3], gsd[m3])
    # per-property, to see which properties the model captures vs misses
    print(f"\n=== BioLiP MODEL | >=3-lig test pockets (n={int(m3.sum())}) ===")
    print(f"  MODEL range IoU: {io3.mean():.3f}")
    print(f"  ---------------------------------------------")
    print(f"  global-avg floor (no features):     0.402")
    print(f"  honest pooled-cluster (geom-blind): 0.403")
    print(f"  oracle (ligand-routed, NOT usable): 0.466")
    print(f"\n  per-property IoU (model):")
    for i, p in enumerate(PROPS):
        ip = iou_range(P[m3][:, [i]], S[m3][:, [i]], gmu[m3][:, [i]], gsd[m3][:, [i]])
        print(f"    {p:16s}{ip.mean():.3f}")


if __name__ == "__main__":
    main()
