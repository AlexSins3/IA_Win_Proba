"""Prédiction d'issue à partir de deux vidéos (P18).

``predict_match(video_a, video_b, checkpoint)`` renvoie un JSON :
    {
      "athlete_a": {"quality_score": ...},
      "athlete_b": {"quality_score": ...},
      "probability_a_wins": ...,
      "probability_b_wins": ...,
      "predicted_winner": "A" | "B",
      "confidence": ...
    }

La cohérence à l'échange est **garantie** : on moyenne les deux ordres, donc
``P(A) + P(B) = 1`` exactement et le résultat est identique si l'on permute les
entrées.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from kata_pipeline.gnn.config import GNNConfig
from kata_pipeline.gnn.data.dataset_hier import compute_in_channels
from kata_pipeline.gnn.models.kata_model import KataOutcomeComparator
from kata_pipeline.gnn.pose.base import build_extractor
from kata_pipeline.gnn.pose.sequence import PoseSequence
from kata_pipeline.gnn.preprocess import preprocess_sequence, sliding_windows
from kata_pipeline.gnn.repro import get_device, load_checkpoint
from kata_pipeline.gnn.skeletons import get_schema

logger = logging.getLogger(__name__)


def _sequence_to_windows(
    seq: PoseSequence,
    config: GNNConfig,
    schema,
    device,
) -> torch.Tensor:
    """Transforme une séquence de poses déjà extraite en tenseur modèle."""

    pre = preprocess_sequence(seq, config.preprocess, schema)
    windows = sliding_windows(pre, config.preprocess)  # (Nw, C, T, J)
    if windows.shape[0] > config.preprocess.max_windows:
        windows = windows[: config.preprocess.max_windows]
    return torch.from_numpy(windows).float().unsqueeze(0).to(device)  # (1, Nw, C, T, J)


def _video_to_sequence(video: str | Path, config: GNNConfig, schema) -> PoseSequence:
    extractor = build_extractor(config.pose, schema)
    return extractor.extract(video)


def _video_to_windows(video: str | Path, config: GNNConfig, schema, device) -> torch.Tensor:
    """Compatibilité interne : extraction puis préparation d'une vidéo."""

    seq = _video_to_sequence(video, config, schema)
    return _sequence_to_windows(seq, config, schema, device)


def predict_sequences(
    sequence_a: PoseSequence,
    sequence_b: PoseSequence,
    checkpoint: str | Path,
    config: GNNConfig,
    debug: bool = False,
) -> dict[str, Any]:
    """Prédit un match depuis deux séquences de poses déjà extraites.

    Cette variante évite de relancer MediaPipe quand l'appelant a besoin des
    poses pour d'autres sorties, par exemple une vidéo avec squelette.
    """

    device = get_device()
    schema = get_schema(config.skeleton.schema_name)
    in_channels = compute_in_channels(config.preprocess)

    model = KataOutcomeComparator(config.model, schema, in_channels).to(device)
    meta = load_checkpoint(checkpoint, model, map_location=str(device))
    ckpt_c = int(meta.get("in_channels", in_channels))
    if ckpt_c != in_channels:
        logger.warning(
            "in_channels config (%d) != checkpoint (%d) : vérifier la cohérence "
            "du prétraitement.", in_channels, ckpt_c
        )
    model.eval()

    xa = _sequence_to_windows(sequence_a, config, schema, device)
    xb = _sequence_to_windows(sequence_b, config, schema, device)
    ma = torch.ones(xa.shape[:2], device=device)
    mb = torch.ones(xb.shape[:2], device=device)

    with torch.no_grad():
        out_ab = model(xa, ma, xb, mb)
        out_ba = model(xb, mb, xa, ma)
        p_b_ab = torch.sigmoid(out_ab["logit"]).item()      # P(B gagne) ordre (A,B)
        p_a_ba = torch.sigmoid(out_ba["logit"]).item()      # P(A gagne) ordre (B,A)
        qa = out_ab["quality_a"].item()
        qb = out_ab["quality_b"].item()

    # Moyenne des deux ordres -> P(A)+P(B)=1 garanti.
    p_a = 0.5 * ((1.0 - p_b_ab) + p_a_ba)
    p_b = 1.0 - p_a
    winner = "A" if p_a >= p_b else "B"
    confidence = abs(p_a - p_b)

    result: dict[str, Any] = {
        "athlete_a": {"quality_score": round(qa, 4)},
        "athlete_b": {"quality_score": round(qb, 4)},
        "probability_a_wins": round(p_a, 4),
        "probability_b_wins": round(p_b, 4),
        "predicted_winner": winner,
        "confidence": round(confidence, 4),
    }
    if debug:
        checkpoint_metrics = meta.get("metrics") or {}
        result["debug"] = {
            "p_b_given_AB": round(p_b_ab, 4),
            "p_a_given_BA": round(p_a_ba, 4),
            "swap_symmetry_error": round(abs(p_b_ab + p_a_ba - 1.0), 5),
            "num_windows_a": int(xa.shape[1]),
            "num_windows_b": int(xb.shape[1]),
            "in_channels": in_channels,
            "schema": schema.name,
            "device": str(device),
            "checkpoint_epoch": meta.get("epoch"),
            "validation_accuracy": checkpoint_metrics.get("accuracy"),
            "validation_roc_auc": checkpoint_metrics.get("roc_auc"),
        }
    return result


def predict_match(
    video_a: str | Path,
    video_b: str | Path,
    checkpoint: str | Path,
    config: GNNConfig,
    debug: bool = False,
) -> dict[str, Any]:
    """Prédit le vainqueur entre deux prestations (anonyme, cohérent à l'échange)."""
    schema = get_schema(config.skeleton.schema_name)
    sequence_a = _video_to_sequence(video_a, config, schema)
    sequence_b = _video_to_sequence(video_b, config, schema)
    return predict_sequences(sequence_a, sequence_b, checkpoint, config, debug=debug)
