"""Boucles d'entraînement pour les deux tâches de la phase 2.

- ``train_comparator`` : la tâche cible ("deux vidéos -> vainqueur").
- ``train_quality``    : tâche auxiliaire (marge de drapeaux + score).

Sur un petit dataset, on privilégie la régularisation (dropout, weight decay),
un split *par match*, et on affiche des métriques honnêtes (accuracy de
comparaison, MAE de marge). Ne pas surinterpréter les résultats : c'est un
premier jet destiné à valider la chaîne complète, pas un juge fiable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from kata_pipeline.gnn.config import GNNConfig
from kata_pipeline.gnn.data.dataset import (
    KataWindowDataset,
    MatchPairDataset,
    build_training_manifest,
    split_manifest,
)
from kata_pipeline.gnn.models.heads import KataComparator, KataQualityNet
from kata_pipeline.gnn.repro import (
    create_run_dir,
    get_device,
    load_checkpoint,
    save_checkpoint,
    save_metrics,
    save_run_config,
    seed_everything,
)

logger = logging.getLogger(__name__)


def train_comparator(
    config: GNNConfig,
    run_name: str = "comparator",
    resume: Optional[Path] = None,
) -> Path:
    """Entraîne le comparateur siamois (prédiction du vainqueur).

    - graines complètes + option déterministe ;
    - split de groupe configurable (``config.train.split_by``) ;
    - run isolé (config résolue + checkpoints + métriques) ;
    - éval honnête : accuracy ordre-agnostique + swap consistency ;
    - reprise possible via ``resume``.
    """

    seed_everything(config.train.seed, config.train.deterministic)
    device = get_device()
    run_dir = create_run_dir(config.paths.runs_dir, run_name)
    save_run_config(config, run_dir)
    logger.info("Run comparateur : %s (device=%s)", run_dir, device)

    manifest = build_training_manifest(config)
    train_df, val_df = split_manifest(
        manifest, config.train.val_ratio, config.train.seed, config.train.split_by
    )

    train_ds = MatchPairDataset(train_df, config, train=True)
    val_ds = MatchPairDataset(val_df, config, train=False)
    if len(train_ds) == 0:
        raise RuntimeError(
            "Aucune paire d'entraînement. Extraire d'abord les poses "
            "(kata-pipeline gnn extract-poses)."
        )

    train_dl = DataLoader(train_ds, batch_size=config.train.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=config.train.batch_size)

    model = KataComparator(config.model).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()

    start_epoch = 0
    best_acc = 0.0
    if resume is not None and Path(resume).exists():
        meta = load_checkpoint(resume, model, optimizer=opt, map_location=str(device))
        start_epoch = int(meta.get("epoch", -1)) + 1
        best_acc = float(meta.get("metrics", {}).get("acc", 0.0))
        logger.info("Reprise depuis %s (epoch %d, best_acc=%.3f)", resume, start_epoch, best_acc)

    best_path = run_dir / "best.pt"
    legacy_path = Path(config.paths.models_dir) / "comparator.pt"
    history: list[dict] = []

    for epoch in range(start_epoch, config.train.epochs):
        model.train()
        total = 0.0
        for batch in train_dl:
            xa = batch["xa"].to(device)
            xb = batch["xb"].to(device)
            y = batch["label"].to(device)
            opt.zero_grad()
            out = model(xa, xb)
            loss = criterion(out["logit"], y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        train_loss = total / max(1, len(train_ds))

        metrics = (
            _eval_comparator(model, val_dl, device)
            if len(val_ds)
            else {"acc": float("nan"), "swap_consistency": float("nan")}
        )
        logger.info(
            "[comparator] epoch %02d | loss=%.4f | val_acc=%.3f | swap_cons=%.3f",
            epoch,
            train_loss,
            metrics["acc"],
            metrics["swap_consistency"],
        )
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})

        improved = len(val_ds) and metrics["acc"] > best_acc
        if improved:
            best_acc = metrics["acc"]
            for path in (best_path, legacy_path):
                save_checkpoint(
                    path,
                    model,
                    config.model.model_dump(),
                    epoch=epoch,
                    optimizer=opt,
                    metrics=metrics,
                )

    if not best_path.exists():  # pas de val -> sauvegarde finale
        for path in (best_path, legacy_path):
            save_checkpoint(path, model, config.model.model_dump(), epoch=config.train.epochs - 1)
    save_metrics({"best_acc": best_acc, "history": history}, run_dir)
    logger.info("Comparateur sauvegardé : %s (best val_acc=%.3f)", best_path, best_acc)
    return best_path


@torch.no_grad()
def _eval_comparator(model: nn.Module, dl: DataLoader, device: torch.device) -> dict[str, float]:
    """Évaluation honnête du comparateur.

    Le dataset de validation fournit toujours ``xa = vainqueur``. On évalue donc
    l'ordre-invariance en présentant les DEUX ordres :
      - ``acc``               : le vrai vainqueur est-il classé au-dessus,
        moyenné sur les deux ordres (élimine tout biais de position) ;
      - ``swap_consistency``  : la prédiction s'inverse-t-elle bien quand on
        échange A et B (garantie d'un juge équitable).
    """

    model.eval()
    correct = consistent = total = 0
    for batch in dl:
        xa = batch["xa"].to(device)  # vainqueur
        xb = batch["xb"].to(device)  # perdant
        p_ab = torch.sigmoid(model(xa, xb)["logit"])  # P(B gagne) avec A=vainqueur
        p_ba = torch.sigmoid(model(xb, xa)["logit"])  # P(A gagne) avec ordre inversé
        # Probabilité ordre-moyennée que le vrai vainqueur (xa) gagne.
        p_winner = 0.5 * ((1.0 - p_ab) + p_ba)
        correct += (p_winner > 0.5).sum().item()
        consistent += ((p_ab > 0.5) != (p_ba > 0.5)).sum().item()
        total += xa.shape[0]
    n = max(1, total)
    return {"acc": correct / n, "swap_consistency": consistent / n}


def train_quality(
    config: GNNConfig,
    run_name: str = "quality",
    resume: Optional[Path] = None,
) -> Path:
    """Entraîne la tête de qualité (marge de drapeaux + score technique).

    Note : ``score`` est généralement absent des CSV (cible entièrement masquée) ;
    seule la marge de drapeaux est réellement apprise. ``flag_result`` étant lié
    au résultat (somme des drapeaux = 5 par match), il s'agit d'un signal de
    *marge de victoire*, auxiliaire au comparateur.
    """

    seed_everything(config.train.seed, config.train.deterministic)
    device = get_device()
    run_dir = create_run_dir(config.paths.runs_dir, run_name)
    save_run_config(config, run_dir)
    logger.info("Run qualité : %s (device=%s)", run_dir, device)

    manifest = build_training_manifest(config)
    train_df, val_df = split_manifest(
        manifest, config.train.val_ratio, config.train.seed, config.train.split_by
    )

    train_ds = KataWindowDataset(train_df, config, train=True)
    val_ds = KataWindowDataset(val_df, config, train=False)
    if len(train_ds) == 0:
        raise RuntimeError("Aucun échantillon. Extraire d'abord les poses.")

    train_dl = DataLoader(train_ds, batch_size=config.train.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=config.train.batch_size)

    model = KataQualityNet(config.model).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )
    mse = nn.MSELoss(reduction="none")

    start_epoch = 0
    best_mae = float("inf")
    if resume is not None and Path(resume).exists():
        meta = load_checkpoint(resume, model, optimizer=opt, map_location=str(device))
        start_epoch = int(meta.get("epoch", -1)) + 1
        best_mae = float(meta.get("metrics", {}).get("flag_mae", float("inf")))
        logger.info("Reprise depuis %s (epoch %d)", resume, start_epoch)

    best_path = run_dir / "best.pt"
    legacy_path = Path(config.paths.models_dir) / "quality.pt"
    history: list[dict] = []

    for epoch in range(start_epoch, config.train.epochs):
        model.train()
        total = 0.0
        for batch in train_dl:
            x = batch["x"].to(device)
            out = model(x)
            loss = _masked_regression_loss(out, batch, mse, config, device)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        train_loss = total / max(1, len(train_ds))
        mae = _eval_quality(model, val_dl, device, config) if len(val_ds) else float("nan")
        logger.info(
            "[quality] epoch %02d | loss=%.4f | val_flag_mae=%.3f", epoch, train_loss, mae
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "flag_mae": mae})
        if len(val_ds) and mae <= best_mae:
            best_mae = mae
            for path in (best_path, legacy_path):
                save_checkpoint(
                    path,
                    model,
                    config.model.model_dump(),
                    epoch=epoch,
                    optimizer=opt,
                    metrics={"flag_mae": mae},
                )

    if not best_path.exists():
        for path in (best_path, legacy_path):
            save_checkpoint(path, model, config.model.model_dump(), epoch=config.train.epochs - 1)
    save_metrics({"best_flag_mae": best_mae, "history": history}, run_dir)
    logger.info("Modèle qualité sauvegardé : %s (best flag MAE=%.3f)", best_path, best_mae)
    return best_path


def _masked_regression_loss(out, batch, mse, config: GNNConfig, device) -> torch.Tensor:
    flag_t = batch["flag"].to(device)
    flag_m = batch["flag_mask"].to(device)
    score_t = batch["score"].to(device)
    score_m = batch["score_mask"].to(device)

    flag_loss = (mse(out["flag"], flag_t) * flag_m).sum() / (flag_m.sum() + 1e-6)
    score_loss = (mse(out["score"], score_t) * score_m).sum() / (score_m.sum() + 1e-6)
    return config.train.w_flag * flag_loss + config.train.w_score * score_loss


@torch.no_grad()
def _eval_quality(model, dl, device, config: GNNConfig) -> float:
    model.eval()
    abs_err = total = 0.0
    scale = config.model.num_flag_classes - 1
    for batch in dl:
        out = model(batch["x"].to(device))
        m = batch["flag_mask"].numpy().astype(bool)
        if m.sum() == 0:
            continue
        pred = out["flag"].cpu().numpy()[m] * scale
        true = batch["flag"].numpy()[m] * scale
        abs_err += np.abs(pred - true).sum()
        total += m.sum()
    return abs_err / max(1, total)
