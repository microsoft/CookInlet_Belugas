"""
Training script for PW_Bioacoustics.

All training parameters are defined in the YAML config file.
Only runtime arguments (split paths, checkpoint) are passed via CLI.

Usage:
    # Train binary from scratch
    python train.py --config config_binary.yaml \\
        --train_csv data/splits_binary/train_split.csv \\
        --val_csv data/splits_binary/val_split.csv \\
        --test_csv data/splits_binary/test_split.csv

    # Train 3-class from scratch
    python train.py --config config_3class.yaml \\
        --train_csv data/splits_3class/train_split.csv \\
        --val_csv data/splits_3class/val_split.csv \\
        --test_csv data/splits_3class/test_split.csv

    # Train 4-class from scratch
    python train.py --config config_4class.yaml \\
        --train_csv data/splits_4class/train_split.csv \\
        --val_csv data/splits_4class/val_split.csv \\
        --test_csv data/splits_4class/test_split.csv

    # Evaluate binary from checkpoint
    python train.py --config config_binary.yaml \\
        --test_csv data/splits_binary/test_split.csv \\
        --ckpt_path checkpoints/binary/best.ckpt

    # Finetune binary from checkpoint
    python train.py --config config_binary.yaml \\
        --train_csv data/splits_binary/train_split.csv \\
        --val_csv data/splits_binary/val_split.csv \\
        --test_csv data/splits_binary/test_split.csv \\
        --ckpt_path checkpoints/binary/best.ckpt --finetune
"""

import argparse
from dataclasses import dataclass
from typing import Optional
import os

import torch
from torch.utils.data import DataLoader
from torchinfo import summary

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Import from PytorchWildlife core library
from PytorchWildlife.models.bioacoustics import ResNetClassifier
from PytorchWildlife.data.bioacoustics.bioacoustics_datasets import (
    BioacousticsDataset,
    SpectrogramAugmentations,
    MixUpCollator,
)
from PytorchWildlife.data.bioacoustics.bioacoustics_configs import load_config


@dataclass
class DataModuleConfig:
    """Configuration for the SpectrogramDataModule."""
    train_csv: Optional[str]
    val_csv: Optional[str]
    test_csv: Optional[str]
    spectrograms_root: str
    x_col: str
    y_col: str
    target_size: list
    batch_size: int
    num_workers: int
    normalize: bool
    pcen: bool
    use_specaug: bool
    use_mixup: bool
    mixup_prob: float
    mixup_alpha: float
    num_classes: Optional[int] = None
    val_spectrograms_root: Optional[str] = None
    test_spectrograms_root: Optional[str] = None
    pin_memory: bool = True
    shuffle_train: bool = True
    # Transform params (used only when use_specaug=True)
    horizontal_shift_prob: float = 0.5
    horizontal_shift_range: float = 0.2
    vertical_shift_prob: float = 0.5
    vertical_shift_range: float = 0.1
    occlusion_prob: float = 0.5
    occlusion_max_lines: int = 3
    occlusion_line_width: float = 0.05
    noise_prob: float = 0.5
    noise_std: float = 0.02
    buffer_prob: float = 0.5
    buffer_max_ratio: float = 0.2
    color_jitter_prob: float = 0.5
    color_jitter_brightness: float = 0.3
    color_jitter_contrast: float = 0.3


class SpectrogramDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for spectrogram classification."""

    def __init__(self, cfg: DataModuleConfig):
        super().__init__()
        self.cfg = cfg
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.eval_transform = None

        if cfg.use_specaug:
            self.train_transform = SpectrogramAugmentations(
                horizontal_shift_prob=self.cfg.horizontal_shift_prob,
                horizontal_shift_range=self.cfg.horizontal_shift_range,
                vertical_shift_prob=self.cfg.vertical_shift_prob,
                vertical_shift_range=self.cfg.vertical_shift_range,
                occlusion_prob=self.cfg.occlusion_prob,
                occlusion_max_lines=self.cfg.occlusion_max_lines,
                occlusion_line_width=self.cfg.occlusion_line_width,
                noise_prob=self.cfg.noise_prob,
                noise_std=self.cfg.noise_std,
                buffer_prob=self.cfg.buffer_prob,
                buffer_max_ratio=self.cfg.buffer_max_ratio,
                color_jitter_prob=self.cfg.color_jitter_prob,
                brightness=self.cfg.color_jitter_brightness,
                contrast=self.cfg.color_jitter_contrast,
            )
        else:
            self.train_transform = None

    def setup(self, stage: Optional[str] = None):
        shared_kwargs = dict(
            x_col=self.cfg.x_col,
            y_col=self.cfg.y_col,
            target_size=self.cfg.target_size,
            normalize=self.cfg.normalize,
        )
        if hasattr(self.cfg, 'pcen'):
            shared_kwargs['pcen'] = self.cfg.pcen
        if self.cfg.num_classes is not None:
            shared_kwargs['num_classes'] = self.cfg.num_classes

        val_root = self.cfg.val_spectrograms_root or self.cfg.spectrograms_root
        test_root = self.cfg.test_spectrograms_root or self.cfg.spectrograms_root

        if self.cfg.train_csv is not None:
            self.train_ds = BioacousticsDataset(
                csv_path=self.cfg.train_csv,
                root=self.cfg.spectrograms_root,
                transform=self.train_transform,
                is_training=True,
                **shared_kwargs
            )
        if self.cfg.val_csv is not None:
            self.val_ds = BioacousticsDataset(
                csv_path=self.cfg.val_csv,
                root=val_root,
                transform=self.eval_transform,
                is_training=False,
                **shared_kwargs
            )
        if self.cfg.test_csv is not None:
            self.test_ds = BioacousticsDataset(
                csv_path=self.cfg.test_csv,
                root=test_root,
                transform=self.eval_transform,
                is_training=False,
                **shared_kwargs
            )

    @property
    def _reference_ds(self):
        return self.train_ds or self.val_ds or self.test_ds

    @property
    def num_classes(self) -> int:
        return self._reference_ds.num_classes

    @property
    def in_channels(self) -> int:
        x0, _, _ = self._reference_ds[0]
        return x0.shape[0]

    @property
    def is_binary(self) -> bool:
        return self.num_classes == 2

    def train_dataloader(self) -> DataLoader:
        if self.is_binary and self.cfg.use_mixup:
            collate_fn = MixUpCollator(
                mixup_prob=self.cfg.mixup_prob,
                mixup_alpha=self.cfg.mixup_alpha
            )
        else:
            collate_fn = None

        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=self.cfg.shuffle_train,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            drop_last=False,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )


def train(cfg, train_csv, val_csv, test_csv, ckpt_path=None, finetune=False,
          predict_only=False, spectrograms_dir=None, val_spectrograms_dir=None,
          test_spectrograms_dir=None, exp_name=None, results_dir="test_results",
          output_csv=None):
    """Train and evaluate the model.
    
    Args:
        cfg: DomainConfig loaded from YAML.
        train_csv: Path to training split CSV.
        val_csv: Path to validation split CSV.
        test_csv: Path to test split CSV.
        ckpt_path: Optional checkpoint to resume/evaluate from.
        finetune: If True and ckpt_path is set, finetune from checkpoint.
        predict_only: If True, skip metrics and only export predictions.
    """
    t = cfg.training

    if spectrograms_dir is not None:
        cfg.paths.spectrograms_dir = spectrograms_dir

    experiment_name = (exp_name or cfg.name) + ("-finetune" if finetune else "")

    dm_cfg = DataModuleConfig(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        spectrograms_root=cfg.paths.spectrograms_dir,
        val_spectrograms_root=val_spectrograms_dir,
        test_spectrograms_root=test_spectrograms_dir,
        x_col=t.x_col,
        y_col=t.y_col,
        target_size=t.target_size,
        batch_size=t.batch_size,
        num_workers=t.num_workers,
        use_specaug=t.use_specaug,
        normalize=t.normalize,
        pcen=getattr(t, 'pcen', False),
        num_classes=t.num_classes if t.num_classes != 2 else None,
        use_mixup=getattr(t, 'use_mixup', False),
        mixup_prob=getattr(t, 'mixup_prob', 0.0),
        mixup_alpha=getattr(t, 'mixup_alpha', 0.2),
    )

    dm = SpectrogramDataModule(dm_cfg)
    dm.setup()

    num_classes = t.num_classes if t.num_classes != 2 else dm.num_classes

    model = ResNetClassifier(
        num_classes=num_classes,
        in_channels=dm.in_channels,
        backbone=t.backbone,
        lr=t.lr,
        weight_decay=t.weight_decay,
        label_smoothing=t.label_smoothing,
        T_max=t.epochs,
        batch_size=t.batch_size,
        pos_weight=t.pos_weight,
        conf_threshold=t.conf_threshold,
        freeze_backbone=getattr(t, 'freeze_backbone', 'none'),
        backbone_lr_ratio=getattr(t, 'backbone_lr_ratio', 1.0),
        class_names=list(cfg.class_names.values()),
    )

    print(f"\nClassification mode: {'Binary' if num_classes == 2 else f'Multiclass ({num_classes} classes)'}")
    print(summary(model, input_size=(t.batch_size, dm.in_channels, *t.target_size)))

    # Callbacks & logging
    monitor_metric = getattr(t, 'monitor_metric', 'val/f1')
    mode = "min" if monitor_metric == "val/loss" else "max"

    eval_only = ckpt_path is not None and not finetune

    ckpt_dir = os.path.join("checkpoints", experiment_name)
    if not eval_only:
        os.makedirs(ckpt_dir, exist_ok=True)

    if not eval_only:
        ckpt_cb = ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor=monitor_metric,
            mode=mode,
            save_top_k=1,
            save_last=True,
            filename="best",
        )
        early_cb = None if finetune else EarlyStopping(monitor=monitor_metric, mode=mode, patience=20)
        trainer_callbacks = [cb for cb in [ckpt_cb, early_cb] if cb is not None]
    else:
        ckpt_cb = None
        early_cb = None
        trainer_callbacks = []

    trainer = pl.Trainer(
        max_epochs=t.epochs,
        accelerator="gpu",
        devices=[0],
        precision="16-mixed",
        gradient_clip_val=1.0,
        log_every_n_steps=20,
        callbacks=trainer_callbacks,
        logger=False,
        enable_progress_bar=True,
    )

    if ckpt_path is None:
        trainer.fit(model, datamodule=dm)
        print(f"Best ckpt: {ckpt_cb.best_model_path}")
        print(f"Best score: {ckpt_cb.best_model_score}")
        if test_csv is not None:
            model.test_csv_path = test_csv
            model.predictions_dir = ckpt_dir
            model.predictions_filename = output_csv
            model.predict_only = predict_only
            test_results = trainer.test(model, datamodule=dm, ckpt_path="best")
        else:
            test_results = []
    else:
        model = ResNetClassifier.load_from_checkpoint(ckpt_path)

        temperature = getattr(t, 'temperature', 1.0)
        if temperature != 1.0:
            model.temperature = torch.tensor(temperature, device=model.device)
            print(f"Using manual temperature: {temperature}")

        if model.is_binary:
            model.hparams.conf_threshold = t.conf_threshold

        if finetune:
            model.hparams.lr = t.lr
            model.hparams.weight_decay = t.weight_decay
            model.hparams.label_smoothing = t.label_smoothing
            model.hparams.T_max = t.epochs
            model.hparams.batch_size = t.batch_size
            model.hparams.freeze_backbone = getattr(t, 'freeze_backbone', 'none')
            model.hparams.backbone_lr_ratio = getattr(t, 'backbone_lr_ratio', 1.0)

            print(f"Finetuning from checkpoint: {ckpt_path}")
            model._apply_freezing_strategy()

            trainer.fit(model, datamodule=dm)
            print("Finetune completed.")
            print(f"Best ckpt: {ckpt_cb.best_model_path}")
            print(f"Best score: {ckpt_cb.best_model_score}")
            if test_csv is not None:
                model.test_csv_path = test_csv
                model.predictions_dir = ckpt_dir
                model.predictions_filename = output_csv
                model.predict_only = predict_only
                test_results = trainer.test(model, datamodule=dm, ckpt_path='best')
            else:
                test_results = []
        else:
            if test_csv is None:
                raise ValueError("--test_csv is required when evaluating from a checkpoint without --finetune")
            test_out_dir = os.path.join(results_dir, experiment_name)
            os.makedirs(test_out_dir, exist_ok=True)
            model.test_csv_path = test_csv
            model.predictions_dir = test_out_dir
            model.predictions_filename = output_csv
            model.predict_only = predict_only
            test_results = trainer.test(model, datamodule=dm)
            print(f"Test completed from checkpoint {ckpt_path}")
            print(f"Results saved to {test_out_dir}")
    
    return test_results[0] if test_results else {}


def main():
    pl.seed_everything(42)

    parser = argparse.ArgumentParser(
        description="Training for bioacoustics classification. "
                    "All model/training params come from the YAML config."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--train_csv", type=str, default=None,
                        help="Path to training split CSV")
    parser.add_argument("--val_csv", type=str, default=None,
                        help="Path to validation split CSV")
    parser.add_argument("--test_csv", type=str, default=None,
                        help="Path to test split CSV (optional; if omitted, testing is skipped)")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Checkpoint to resume/evaluate from")
    parser.add_argument("--finetune", action="store_true",
                        help="Finetune from --ckpt_path instead of evaluating")
    parser.add_argument("--predict_only", action="store_true",
                        help="Skip metrics, only export predictions CSV. "
                             "Useful when test labels are in a different "
                             "label space than the model.")
    parser.add_argument("--spectrograms_dir", type=str, default=None,
                        help="Override spectrograms_dir from config (applies to train split)")
    parser.add_argument("--val_spectrograms_dir", type=str, default=None,
                        help="Override spectrograms_dir for val split (falls back to --spectrograms_dir)")
    parser.add_argument("--test_spectrograms_dir", type=str, default=None,
                        help="Override spectrograms_dir for test split (falls back to --spectrograms_dir)")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Override cfg.name for output directory naming")
    parser.add_argument("--results_dir", type=str, default="test_results",
                        help="Directory for test-only CSV results (default: test_results)")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Override the output CSV filename (e.g. my_results.csv). "
                             "If omitted, defaults to <test_csv_basename>_with_predictions.csv")

    args = parser.parse_args()

    CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
    config_path = args.config
    if not os.path.isabs(config_path) and not os.path.exists(config_path):
        config_path = os.path.join(CONFIGS_DIR, config_path)
    cfg = load_config(config_path)

    train(
        cfg=cfg,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        ckpt_path=args.ckpt_path,
        finetune=args.finetune,
        predict_only=args.predict_only,
        spectrograms_dir=args.spectrograms_dir,
        val_spectrograms_dir=args.val_spectrograms_dir,
        test_spectrograms_dir=args.test_spectrograms_dir,
        exp_name=args.exp_name,
        results_dir=args.results_dir,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
