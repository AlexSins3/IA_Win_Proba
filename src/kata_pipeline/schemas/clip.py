"""Schéma du dataset final de clips kata."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, computed_field, field_validator

from kata_pipeline.competition_formats import CompetitionType


class KataClip(BaseModel):
    """Un clip vidéo de kata extrait et aligné avec les métadonnées sportives."""

    id_clip: str
    id_match: str
    id_live: str
    video_url: str | None = None
    source_video_path: Path | None = None
    clip_path: Path | None = None
    athlete: str
    color: Literal["red", "blue"]
    kata: str
    style: str | None = None
    score: float | None = None
    flag_result: str | None = None
    issue: Literal["win", "loss", "unknown"] = "unknown"
    opponent: str
    opponent_kata: str
    competition: str
    competition_type: CompetitionType | None = None
    category: str
    round: str
    match_order: int
    passage_order: int
    start_time: float  # en secondes
    end_time: float  # en secondes
    confidence_score: float = 0.0
    validation_status: Literal[
        "auto_validated", "manually_validated", "needs_correction", "rejected", "pending"
    ] = "pending"
    needs_review: bool = False

    @field_validator("clip_path", "source_video_path", "competition_type", mode="before")
    @classmethod
    def _coerce_nan_to_none(cls, v: object) -> object:
        """Convertir NaN (venant de pandas CSV) en None."""
        if v is None:
            return None
        if isinstance(v, float) and math.isnan(v):
            return None
        if isinstance(v, str) and v.lower() == "nan":
            return None
        return v

    @field_validator("flag_result", mode="before")
    @classmethod
    def _coerce_flag_result(cls, v: object) -> object:
        """Convertir int ou float en string pour flag_result."""
        if v is None:
            return None
        if isinstance(v, float):
            if math.isnan(v):
                return None
            return str(int(v))
        if isinstance(v, int):
            return str(v)
        return v

    @field_validator("score", mode="before")
    @classmethod
    def _coerce_score_nan(cls, v: object) -> object:
        """Convertir NaN en None pour score."""
        if isinstance(v, float) and math.isnan(v):
            return None
        return v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration(self) -> float:
        """Durée du clip en secondes."""
        return self.end_time - self.start_time
