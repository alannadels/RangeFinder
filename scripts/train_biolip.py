"""
BioLiP integration -- Stage 4: train on the BioLiP-enriched dataset.
Same GIoU recipe as the best general model (eval_0043), pointed at the BioLiP
point clouds + enriched pocket distributions.

    python -m scripts.train_biolip [num_epochs]
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.config import Config
from model import train

BL = PROJECT_ROOT / "data" / "biolip"
LOSS_WEIGHTS = [
    0.5866810037922697, 0.28300598427959284, 1.4322480566305442, 4.148594786142039,
    0.7502064398195392, 0.754906675747274, 1.3726759557083994, 0.52043057786674,
    0.30943314903605645, 0.5191008838523781, 0.4263960123904286, 0.23567080205409147,
    2.6641871724772774, 0.2356861484999496, 6.902619486165242, 0.40188877225309483,
    0.8491578688623318, 2.3597024241570983,
]


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    cfg = Config(
        use_iou_loss=True,
        loss_weights=LOSS_WEIGHTS,
        train_csv=BL / "train.csv",
        val_csv=BL / "val.csv",
        test_csv=BL / "test.csv",
        pocket_distributions_csv=BL / "pilot_pocket_distributions.csv",
        pointcloud_dir=BL / "pointclouds_esm",
        sigma_norm_min_ligands=3,
        learning_rate=0.000223250117360305,
        weight_decay=0.0001,
        head_dropout=0.244855281994370,
        trunk_dropout=0.234614637776889,
        atom_dropout_rate=0.0,
        grad_clip_norm=3.971850615199632,
        aromatic_bce_weight=0.088867,
        batch_size=2,          # ignored when atom_budget is set (kept as a fallback)
        atom_budget=80000,     # dynamic batching by EDGE count (memory driver, not atoms):
                               # pack <=80k edges/batch. typical pocket ~23k edges -> ~3/batch;
                               # dense ones batch alone. bounds GPU memory, keeps ~all pockets.
        seed=42,
        lr_patience=10,        # decay LR after 10 epochs w/o val improvement
        num_epochs=epochs,
        early_stop_patience=12,  # stop after 12 epochs w/o val improvement
        checkpoint_dir=PROJECT_ROOT / "model" / "checkpoints" / "biolip_pilot",
    )
    print(f"=== Training on BioLiP-enriched pilot ({epochs} epochs) ===")
    print(f"  pointclouds: {cfg.pointcloud_dir}")
    print(f"  distributions: {cfg.pocket_distributions_csv}")
    train.main(cfg)


if __name__ == "__main__":
    main()
