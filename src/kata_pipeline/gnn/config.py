"""Configuration centralisée pour la phase 2 (poses + GNN).

Toutes les valeurs ont des défauts raisonnables pour un *premier jet* sur un
petit dataset. Elles sont surchargeables via un YAML passé à ``load_gnn_config``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

from kata_pipeline.competition_formats import CompetitionType


class PoseConfig(BaseModel):
    """Paramètres d'extraction de poses (MediaPipe)."""

    target_fps: float = Field(
        10.0,
        description="Ré-échantillonnage temporel des poses (Hz). 10 fps suffit "
        "pour capturer la dynamique d'un kata tout en restant léger.",
    )
    model_complexity: int = Field(
        1,
        ge=0,
        le=2,
        description="0=lite (rapide), 1=full (équilibré, défaut), 2=heavy (précis "
        "mais lent sur CPU). Mappé sur les modèles PoseLandmarker .task.",
    )
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    # On garde un sous-ensemble stable de 15 articulations (voir skeleton.py).
    use_subset: bool = True
    # Extracteur de pose : "mediapipe" (défaut) ou "mmpose".
    extractor: str = "mediapipe"
    device: str = "auto"  # auto|cpu|cuda (MMPose)
    batch_size: int = 1   # batch d'inférence (MMPose)
    # Poids MMPose (aucun téléchargement silencieux : chemins explicites requis).
    mmpose_pose_config: Optional[str] = None
    mmpose_checkpoint: Optional[str] = None
    mmpose_det_config: Optional[str] = None
    mmpose_det_checkpoint: Optional[str] = None


class GraphConfig(BaseModel):
    """Construction du graphe spatio-temporel."""

    window_size: int = Field(
        120,
        description="Nombre de frames de pose par échantillon (à target_fps). "
        "120 @ 10fps = 12 s ; un kata dure ~1-3 min, on échantillonne des fenêtres.",
    )
    window_stride: int = Field(60, description="Pas entre deux fenêtres (recouvrement).")
    normalize: bool = Field(
        True,
        description="Centrer sur le bassin + normaliser par la taille du torse "
        "pour rendre le modèle invariant à la position/échelle dans l'image.",
    )


class SkeletonConfig(BaseModel):
    """Choix du schéma d'articulations (voir gnn/skeletons.py)."""

    schema_name: str = Field(
        "mediapipe_15",
        description="mediapipe_15 (baseline) | mediapipe_33 | canonical_body…",
    )


class PreprocessConfig(BaseModel):
    """Prétraitement robuste et features biomécaniques (pipeline hiérarchique)."""

    # Points manquants.
    confidence_threshold: float = 0.3
    max_interp_gap_seconds: float = 0.5
    smoothing: str = "none"  # none|moving_avg
    smoothing_window: int = 5
    # Normalisation : none|torso|bbox ; rotation d'alignement (destructive) off.
    normalization: str = "torso"
    rotate_align: bool = False
    # Features.
    use_z: bool = True
    streams: list[str] = Field(
        default_factory=lambda: ["joint", "joint_motion"],
        description="joint | joint_motion | bone | bone_motion",
    )
    use_acceleration: bool = False
    use_angles: bool = True
    use_confidence: bool = True
    use_masks: bool = False
    # Fenêtres définies en secondes (converties en frames via target_fps).
    window_duration_seconds: float = 12.0
    window_stride_seconds: float = 6.0
    max_windows: int = 16


class ModelConfig(BaseModel):
    """Hyperparamètres du ST-GCN."""

    in_channels: int = Field(
        9,
        description="Canaux par nœud. Recalculé automatiquement pour la pipeline "
        "hiérarchique à partir de preprocess (streams/options).",
    )
    hidden_channels: int = 64
    embedding_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.3
    num_flag_classes: int = 6  # drapeaux 0..5 (WKF)
    # Encodeur spatio-temporel : "stgcn" | "multistream_stgcn" | "ctrgcn".
    encoder: str = "stgcn"
    # Agrégateur des fenêtres -> prestation : "mean" (baseline) | "attention".
    aggregator: str = "mean"
    # Résidu contextuel antisymétrique optionnel dans le comparateur.
    context_residual: bool = False
    # Nombre max de fenêtres agrégées par prestation (padding/troncature).
    max_windows: int = 16


class TrainConfig(BaseModel):
    """Hyperparamètres d'entraînement."""

    batch_size: int = 8
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_ratio: float = 0.2
    seed: int = 42
    # None = entraîner conjointement sur tous les circuits disponibles.
    competition_type: CompetitionType | None = None
    # Regroupement pour le split train/val (anti-fuite).
    #   "match"       : deux katas d'un même combat restent ensemble (baseline).
    #   "athlete"     : test "athlètes non vus" (les matchs à cheval sont écartés).
    #   "competition" : test "compétition/source non vue".
    split_by: str = "match"
    # Force les algorithmes déterministes (cuDNN) pour une reproductibilité stricte.
    deterministic: bool = False
    # Pondération des différentes têtes dans la perte multi-tâches.
    w_outcome: float = 1.0
    w_flag: float = 0.5
    w_score: float = 0.5


class PathsConfig(BaseModel):
    """Emplacements sur disque."""

    clips_dir: Path = Path("data/clips")
    output_dir: Path = Path("data/output")
    poses_dir: Path = Path("data/poses")
    models_dir: Path = Path("models")
    viz_dir: Path = Path("data/viz")
    # Dossier d'isolation des runs (config résolue + checkpoints + métriques).
    runs_dir: Path = Path("runs")


class ArchiveConfig(BaseModel):
    """Archive vidéo Google Drive et cache local à la demande."""

    enabled: bool = False
    root_folder_id: Optional[str] = None
    remote_clips_path: str = "clips"
    credentials_file: Path = Path(".secrets/google_drive_credentials.json")
    token_file: Path = Path(".secrets/google_drive_token.json")
    index_file: Path = Path("data/archive/drive_videos.csv")
    cache_dir: Path = Path("data/cache/videos")
    cache_max_gb: float = Field(10.0, gt=0)
    verify_md5: bool = True


class GNNConfig(BaseModel):
    """Configuration complète de la phase 2."""

    pose: PoseConfig = PoseConfig()
    graph: GraphConfig = GraphConfig()
    skeleton: SkeletonConfig = SkeletonConfig()
    preprocess: PreprocessConfig = PreprocessConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    paths: PathsConfig = PathsConfig()
    archive: ArchiveConfig = ArchiveConfig()


def load_gnn_config(path: str | Path | None = None) -> GNNConfig:
    """Charge la config, éventuellement surchargée par un fichier YAML."""

    if path is None:
        return GNNConfig()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return GNNConfig(**data)
