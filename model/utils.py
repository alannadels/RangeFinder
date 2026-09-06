"""
Utility functions: seeding, checkpointing, logging.
"""

import os
import random
import json
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import torch


def seed_everything(seed):
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic operations (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, config,
                    target_mean, target_std, path,
                    geom_mean=None, geom_std=None,
                    cloud_mean=None, cloud_std=None):
    """Save a full training checkpoint for reproducible resumption.

    Saves everything needed to resume training or run inference:
    model weights, optimizer state, scheduler state, epoch, validation loss,
    config, and target normalization stats. When the geometric auxiliary task
    is enabled, also stores geom_mean / geom_std for eval-time denormalization.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "target_mean": target_mean,
        "target_std": target_std,
        "geom_mean": geom_mean,
        "geom_std": geom_std,
        "use_geometric_targets": getattr(config, "use_geometric_targets", False),
        "cloud_mean": cloud_mean,
        "cloud_std": cloud_std,
        "use_exemplar_head": getattr(config, "use_exemplar_head", False),
        "n_exemplars": getattr(config, "n_exemplars", None),
        "use_mixture_heads": getattr(config, "use_mixture_heads", False),
        "config": {
            "hidden_scalars": config.hidden_scalars,
            "hidden_vectors": config.hidden_vectors,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "lmax": config.lmax,
            "radius_cutoff": config.radius_cutoff,
            "num_radial_basis": config.num_radial_basis,
            "radial_mlp_hidden": config.radial_mlp_hidden,
            "head_hidden_1": config.head_hidden_1,
            "head_hidden_2": config.head_hidden_2,
            "head_dropout": config.head_dropout,
            "input_features": config.input_features,
            "num_targets": config.num_targets,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "seed": config.seed,
        },
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Load a checkpoint. Returns the checkpoint dict.

    If optimizer and scheduler are provided, their states are restored too.
    """
    device = next(model.parameters()).device
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


class TrainingLogger:
    """Logs training metrics to console and a CSV file."""

    def __init__(self, log_dir):
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = log_dir / f"training_log_{timestamp}.csv"
        self.start_time = datetime.now()

        # Write CSV header
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "val_loss", "learning_rate",
                "elapsed_minutes", "best_val_loss",
            ])

    def log_epoch(self, epoch, train_loss, val_loss, lr, best_val_loss):
        """Log one epoch's metrics."""
        elapsed = (datetime.now() - self.start_time).total_seconds() / 60.0

        # Console output
        print(
            f"Epoch {epoch:3d} | "
            f"Train {train_loss:.6f} | "
            f"Val {val_loss:.6f} | "
            f"LR {lr:.2e} | "
            f"Best {best_val_loss:.6f} | "
            f"{elapsed:.1f}min"
        )

        # CSV output
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                f"{lr:.2e}", f"{elapsed:.1f}", f"{best_val_loss:.6f}",
            ])

    def log_message(self, msg):
        """Log a general message to console."""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
