"""Schéma du dataset de compétition."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator

from kata_pipeline.competition_formats import CompetitionType


class MatchRecord(BaseModel):
    """Un match de kata dans le dataset de compétition."""

    competition: str
    competition_type: CompetitionType | None = None
    category: str
    round: str
    match_order: int
    athlete_red: str
    athlete_blue: str
    kata_red: str
    kata_blue: str
    style_red: str | None = None
    style_blue: str | None = None
    score_red: float | None = None
    score_blue: float | None = None
    flag_result_red: str | None = None
    flag_result_blue: str | None = None
    winner: str | None = None
    round_outcome: str | None = None
    decision_type: Literal["score", "flag", "unknown"] | None = None
    section: str | None = None  # sous-section du live (pool_1, pool_7, etc.)

    @field_validator("flag_result_red", "flag_result_blue", mode="before")
    @classmethod
    def coerce_flag_result(cls, v: object) -> str | None:
        if v is None:
            return None
        return str(v)

    @field_validator("match_order", mode="before")
    @classmethod
    def coerce_match_order(cls, v: int | str) -> int:
        return int(v)


# Colonnes minimales requises dans le dataset CSV/Excel
REQUIRED_COLUMNS: list[str] = [
    "competition",
    "category",
    "round",
    "match_order",
    "athlete_red",
    "athlete_blue",
    "kata_red",
    "kata_blue",
]

# Colonnes optionnelles reconnues
OPTIONAL_COLUMNS: list[str] = [
    "competition_type",
    "section",
    "style_red",
    "style_blue",
    "score_red",
    "score_blue",
    "flag_result_red",
    "flag_result_blue",
    "winner",
    "round_outcome",
    "decision_type",
]
