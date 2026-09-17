"""Schéma des métadonnées de lives."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from kata_pipeline.competition_formats import CompetitionType


class LiveRecord(BaseModel):
    """Métadonnées d'un live YouTube."""

    id_live: str
    url: str | None = None
    local_path: Path | None = None
    competition: str
    competition_type: CompetitionType | None = None
    category: str
    section: str | None = None  # sous-section du live (pool_1, pool_7, repechage_1...)
    useful_start: float | None = None  # en secondes
    useful_end: float | None = None  # en secondes
    analysis_quality: Literal["low", "medium", "high"] = "low"
    clip_quality: Literal["low", "medium", "high"] = "high"
    status: Literal["pending", "downloaded", "analyzed", "completed", "error"] = "pending"


REQUIRED_LIVE_COLUMNS: list[str] = [
    "id_live",
    "competition",
    "category",
]

OPTIONAL_LIVE_COLUMNS: list[str] = [
    "competition_type",
    "url",
    "local_path",
    "section",
    "useful_start",
    "useful_end",
    "analysis_quality",
    "clip_quality",
    "status",
]
