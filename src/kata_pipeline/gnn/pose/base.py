"""Interface commune des extracteurs de pose et fabrique (P6).

Tout extracteur implémente ``PoseExtractor.extract(video) -> PoseSequence`` et
gère son propre cache. On sélectionne l'implémentation via la configuration
(``pose.extractor``), ce qui rend MediaPipe et MMPose interchangeables sans
toucher au reste de la pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from kata_pipeline.gnn.config import PoseConfig
from kata_pipeline.gnn.pose.sequence import PoseSequence
from kata_pipeline.gnn.skeletons import SkeletonSchema


@runtime_checkable
class PoseExtractor(Protocol):
    """Contrat minimal d'un extracteur de pose."""

    def extract(self, video_path: str | Path) -> PoseSequence:
        """Extrait la séquence de poses d'une vidéo."""
        ...


def build_extractor(config: PoseConfig, schema: SkeletonSchema) -> PoseExtractor:
    """Fabrique l'extracteur demandé par la config.

    ``pose.extractor`` : ``"mediapipe"`` (défaut) ou ``"mmpose"``.
    """

    name = (config.extractor or "mediapipe").lower()
    if name == "mediapipe":
        from kata_pipeline.gnn.pose.mediapipe_extractor import MediaPipePoseExtractor

        return MediaPipePoseExtractor(config, schema)
    if name == "mmpose":
        from kata_pipeline.gnn.pose.mmpose_backend import MMPosePoseExtractor

        return MMPosePoseExtractor(config, schema)
    raise ValueError(
        f"Extracteur de pose inconnu : '{config.extractor}'. Attendu : 'mediapipe' ou 'mmpose'."
    )
