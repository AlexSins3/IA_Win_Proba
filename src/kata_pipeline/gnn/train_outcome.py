"""Entraînement de la tâche d'issue avec le modèle hiérarchique (P14/P16).

Chemin *opt-in* : n'altère pas ``train_comparator`` (baseline mono-fenêtre). Il
utilise ``PerformancePairDataset`` (toutes les fenêtres) + ``KataOutcomeComparator``
(antisymétrique) et rapporte des métriques honnêtes ordre-invariantes.
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from kata_pipeline.gnn.augment import AugmentConfig
from kata_pipeline.gnn.config import GNNConfig
from kata_pipeline.gnn.data.dataset import build_training_manifest, split_manifest
from kata_pipeline.gnn.data.dataset_hier import (
    PerformancePairDataset,
    collate_pairs,
    compute_in_channels,
)
from kata_pipeline.gnn.evaluate import outcome_metrics
from kata_pipeline.gnn.models.kata_model import KataOutcomeComparator
from kata_pipeline.gnn.repro import (
    create_run_dir,
    get_device,
    load_checkpoint,
    save_checkpoint,
    save_metrics,
    save_run_config,
    seed_everything,
)
from kata_pipeline.gnn.skeletons import get_schema

logger = logging.getLogger(__name__)


def _bone_split(config: GNNConfig, in_channels: int) -> int | None:
    """Point de coupe des canaux pour l'encodeur multi-flux (joint vs bone)."""

    if config.model.encoder != "multistream_stgcn":
        return None
    d = 3 if config.preprocess.use_z else 2
    joint_ch = 0
    if "joint" in config.preprocess.streams:
        joint_ch += d
    if "joint_motion" in config.preprocess.streams:
        joint_ch += d
    return joint_ch if 0 < joint_ch < in_channels else in_channels // 2


def train_outcome(
    config: GNNConfig,
    run_name: str = "outcome",
    resume: Optional[Path] = None,
) -> Path:
    """Entraîne le comparateur d'issue hiérarchique."""

    seed_everything(config.train.seed, config.train.deterministic)
    device = get_device()
    run_dir = create_run_dir(config.paths.runs_dir, run_name)
    save_run_config(config, run_dir)

    schema = get_schema(config.skeleton.schema_name)
    in_channels = compute_in_channels(config.preprocess)
    logger.info("Run issue : %s | schéma=%s | C=%d | device=%s",
                run_dir, schema.name, in_channels, device)

    manifest = build_training_manifest(config)
    train_df, val_df = split_manifest(
        manifest, config.train.val_ratio, config.train.seed, config.train.split_by
    )
    aug = AugmentConfig()
    train_ds = PerformancePairDataset(train_df, config, train=True, augment=aug)
    val_ds = PerformancePairDataset(val_df, config, train=False)
    if len(train_ds) == 0:
        raise RuntimeError("Aucune paire d'entraînement (extraire les poses d'abord).")

    collate = partial(collate_pairs, max_windows=config.preprocess.max_windows)
    train_dl = DataLoader(
        train_ds,
        batch_size=config.train.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_dl = DataLoader(val_ds, batch_size=config.train.batch_size, collate_fn=collate)

    model = KataOutcomeComparator(
        config.model, schema, in_channels, bone_split=_bone_split(config, in_channels)
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()

    start_epoch, best_acc = 0, 0.0
    if resume is not None and Path(resume).exists():
        meta = load_checkpoint(resume, model, optimizer=opt, map_location=str(device))
        start_epoch = int(meta.get("epoch", -1)) + 1
        best_acc = float(meta.get("metrics", {}).get("accuracy", 0.0))

    best_path = run_dir / "best.pt"
    legacy_path = Path(config.paths.models_dir) / "outcome.pt"
    history: list[dict] = []

    for epoch in range(start_epoch, config.train.epochs):
        model.train()
        total = 0.0
        for batch in train_dl:
            xa, ma = batch["xa"].to(device), batch["mask_a"].to(device)
            xb, mb = batch["xb"].to(device), batch["mask_b"].to(device)
            y = batch["label"].to(device)
            opt.zero_grad()
            out = model(xa, ma, xb, mb)
            loss = criterion(out["logit"], y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        train_loss = total / max(1, len(train_ds))

        metrics = (
            _eval_outcome(model, val_dl, device)
            if len(val_ds)
            else {"accuracy": float("nan")}
        )
        logger.info(
            "[outcome] epoch %02d | loss=%.4f | val_acc=%.3f | auc=%.3f | swap_cons=%.3f",
            epoch, train_loss, metrics.get("accuracy", float("nan")),
            metrics.get("roc_auc", float("nan")), metrics.get("swap_consistency", float("nan")),
        )
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})

        acc = metrics.get("accuracy", float("nan"))
        if len(val_ds) and not np.isnan(acc) and acc > best_acc:
            best_acc = acc
            for path in (best_path, legacy_path):
                save_checkpoint(path, model, config.model.model_dump(), epoch=epoch,
                                optimizer=opt, metrics=metrics,
                                extra={"in_channels": in_channels, "schema": schema.name})

    if not best_path.exists():
        for path in (best_path, legacy_path):
            save_checkpoint(path, model, config.model.model_dump(), epoch=config.train.epochs - 1,
                            extra={"in_channels": in_channels, "schema": schema.name})
    save_metrics({"best_acc": best_acc, "history": history}, run_dir)
    logger.info("Modèle d'issue sauvegardé : %s (best val_acc=%.3f)", best_path, best_acc)
    return best_path


@torch.no_grad()
def _eval_outcome(model: nn.Module, dl: DataLoader, device: torch.device) -> dict:
    """Éval honnête : présente les deux ordres et agrège les métriques d'issue.

    Le dataset de val fournit toujours A=vainqueur (label 0). On calcule la
    probabilité que B gagne dans l'ordre donné et dans l'ordre échangé.
    """

    model.eval()
    y_true, p_b, p_b_sw = [], [], []
    for batch in dl:
        xa, ma = batch["xa"].to(device), batch["mask_a"].to(device)
        xb, mb = batch["xb"].to(device), batch["mask_b"].to(device)
        y = batch["label"].cpu().numpy()
        p_ab = torch.sigmoid(model(xa, ma, xb, mb)["logit"]).cpu().numpy()
        p_ba = torch.sigmoid(model(xb, mb, xa, ma)["logit"]).cpu().numpy()
        y_true.append(y)
        p_b.append(p_ab)          # P(B gagne) ordre (A,B)
        p_b_sw.append(p_ba)       # P(A gagne) ordre (B,A) == P(B gagne) après échange
    y_true = np.concatenate(y_true)
    p_b = np.concatenate(p_b)
    p_b_sw = np.concatenate(p_b_sw)
    return outcome_metrics(y_true, p_b, p_b_swapped=p_b_sw)
