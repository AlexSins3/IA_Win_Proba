"""Stub d'extracteur MMPose (P6/P8).

Interface conforme à ``PoseExtractor`` mais **volontairement non fonctionnelle
sans poids fournis**. Aucun téléchargement silencieux : les chemins de config et
de checkpoint doivent être renseignés explicitement (``pose.mmpose_*``). Cela
permet de brancher un extracteur plus précis (whole-body, mains/pieds) le jour
où l'on dispose des poids, sans réécrire le reste de la pipeline.
"""

from __future__ import annotations

from pathlib import Path

from kata_pipeline.gnn.config import PoseConfig
from kata_pipeline.gnn.pose.sequence import PoseSequence
from kata_pipeline.gnn.skeletons import SkeletonSchema


class MMPosePoseExtractor:
    """Extracteur MMPose (à activer avec des poids locaux)."""

    def __init__(self, config: PoseConfig, schema: SkeletonSchema):
        self.config = config
        self.schema = schema
        self._check_weights()
        self._model = None  # chargé paresseusement dans _ensure_model()

    def _check_weights(self) -> None:
        missing = [
            name
            for name, val in (
                ("pose.mmpose_pose_config", self.config.mmpose_pose_config),
                ("pose.mmpose_checkpoint", self.config.mmpose_checkpoint),
            )
            if not val
        ]
        if missing:
            raise ValueError(
                "Extracteur MMPose sélectionné mais poids/config manquants : "
                + ", ".join(missing)
                + ". Renseignez ces chemins locaux dans la config (aucun "
                "téléchargement automatique n'est effectué)."
            )
        for name, val in (
            ("mmpose_pose_config", self.config.mmpose_pose_config),
            ("mmpose_checkpoint", self.config.mmpose_checkpoint),
        ):
            if val and not Path(val).exists():
                raise FileNotFoundError(f"Fichier MMPose introuvable ({name}) : {val}")

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from mmpose.apis import init_model  # type: ignore
        except ImportError as exc:  # pragma: no cover - dépendance optionnelle
            raise ImportError(
                "Le paquet 'mmpose' n'est pas installé. Installez-le pour utiliser "
                "l'extracteur MMPose (extra optionnel)."
            ) from exc
        device = self.config.device if self.config.device != "auto" else None
        self._model = init_model(
            self.config.mmpose_pose_config,
            self.config.mmpose_checkpoint,
            device=device,
        )

    def extract(self, video_path: str | Path) -> PoseSequence:  # pragma: no cover - stub
        self._ensure_model()
        raise NotImplementedError(
            "L'inférence MMPose n'est pas encore implémentée dans ce stub. "
            "Le point d'entrée (config, vérification des poids, sélection du "
            "device) est prêt : brancher ici la boucle d'inférence MMPose et "
            "mapper la sortie vers le schéma via schema.source_indices."
        )
