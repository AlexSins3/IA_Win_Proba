"""Configuration de la pipeline kata."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class VideoConfig(BaseModel):
    analysis_width: int = 480
    analysis_fps: int = 2
    clip_quality: Literal["high", "medium", "low"] = "high"
    clip_margin_before: float = 3.0
    clip_margin_after: float = 3.0


class MotionConfig(BaseModel):
    blur_kernel_size: int = 21
    smoothing_window: float = 3.0
    activity_threshold: float = 15.0


class SegmentsConfig(BaseModel):
    min_duration: float = 90.0
    max_duration: float = 280.0
    min_motion_score: float = 10.0
    merge_gap: float = 20.0
    auto_validate_confidence: float = 0.8
    kata_durations_ref: Path | None = None


class PairingConfig(BaseModel):
    max_gap_within_match: float = 120.0
    min_gap_between_matches: float = 30.0
    duration_ratio_tolerance: float = 0.3


class PathsConfig(BaseModel):
    input_dir: Path = Path("data/input")
    intermediate_dir: Path = Path("data/intermediate")
    # Chemins modèles : le pipeline insère SA/K1 avant le dernier segment.
    clips_pending_dir: Path = Path("data/clips/pending")
    clips_validated_dir: Path = Path("data/clips/validated")
    output_dir: Path = Path("data/output")
    validation_file: Path = Path("data/output/validation.csv")
    backup_dir: Path = Path("data/backups")


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: Path = Path("logs/pipeline.log")
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


class PipelineConfig(BaseSettings):
    video: VideoConfig = VideoConfig()
    motion: MotionConfig = MotionConfig()
    segments: SegmentsConfig = SegmentsConfig()
    pairing: PairingConfig = PairingConfig()
    paths: PathsConfig = PathsConfig()
    logging: LoggingConfig = LoggingConfig()

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """Charger la configuration depuis un fichier YAML."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data) if data else cls()

    @classmethod
    def load(cls, config_path: Path | None = None) -> "PipelineConfig":
        """Charger la configuration avec fallback sur le fichier par défaut."""
        if config_path and config_path.exists():
            return cls.from_yaml(config_path)

        default_path = Path("config/default.yaml")
        if default_path.exists():
            return cls.from_yaml(default_path)

        return cls()

    def ensure_directories(self) -> None:
        """Créer tous les répertoires nécessaires."""
        for field_name in self.paths.model_fields:
            path = getattr(self.paths, field_name)
            if field_name.endswith("_dir"):
                if field_name in {"clips_pending_dir", "clips_validated_dir"}:
                    path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)

        self.logging.file.parent.mkdir(parents=True, exist_ok=True)
