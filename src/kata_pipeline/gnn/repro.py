"""Reproductibilité et gestion des runs pour la phase 2.

Regroupe tout ce qui rend un entraînement rejouable et traçable :
- ``seed_everything`` : graine unique pour ``random``, NumPy et PyTorch (+CUDA),
  avec option d'algorithmes déterministes (cuDNN) ;
- ``get_device`` : CPU/CUDA ;
- ``create_run_dir`` / ``save_run_config`` : isolation d'un run (config résolue
  sauvegardée à côté des checkpoints et des métriques) ;
- ``save_checkpoint`` / ``load_checkpoint`` : format de checkpoint unique,
  rétro-compatible (clés ``model`` + ``config``) et enrichi (epoch, métriques,
  état de l'optimiseur) pour permettre la reprise.
"""

from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from kata_pipeline.gnn.config import GNNConfig

logger = logging.getLogger(__name__)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Fixe toutes les sources d'aléa pour un run reproductible."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # noqa: BLE001 - selon build PyTorch
            logger.debug("use_deterministic_algorithms indisponible sur ce build.")


def seed_worker(worker_id: int) -> None:
    """Graine par worker DataLoader (à passer via ``worker_init_fn``)."""

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device() -> torch.device:
    """Retourne CUDA si disponible, sinon CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_run_dir(runs_dir: str | Path, name: str) -> Path:
    """Crée ``runs_dir/<name>_<timestamp>`` et le retourne."""

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(runs_dir) / f"{name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_config(config: GNNConfig, run_dir: Path) -> Path:
    """Sauvegarde la config résolue (YAML) dans le dossier du run."""

    path = Path(run_dir) / "config.yaml"
    payload = config.model_dump(mode="json")  # convertit les Path en str
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def save_metrics(metrics: dict[str, Any], run_dir: Path, name: str = "metrics.json") -> Path:
    """Sauvegarde un dictionnaire de métriques en JSON."""

    path = Path(run_dir) / name
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    model_config: dict[str, Any],
    *,
    epoch: int | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Écrit un checkpoint rétro-compatible (clés ``model`` + ``config``)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "config": model_config,
    }
    if epoch is not None:
        payload["epoch"] = epoch
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if metrics is not None:
        payload["metrics"] = metrics
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Recharge un checkpoint (poids, et optionnellement l'optimiseur).

    Retourne les métadonnées (``epoch``, ``metrics``…) présentes dans le fichier.
    """

    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return {k: v for k, v in ckpt.items() if k not in {"model", "optimizer"}}
