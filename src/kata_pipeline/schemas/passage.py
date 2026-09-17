"""Schéma des passages attendus (un par athlète par match)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from kata_pipeline.competition_formats import CompetitionType


class ExpectedPassage(BaseModel):
    """Un passage de kata attendu (rouge ou bleu)."""

    id_match: str
    id_live: str
    competition: str
    competition_type: CompetitionType | None = None
    category: str
    round: str
    match_order: int
    passage_order: int  # 1 = rouge, 2 = bleu dans un match
    color: Literal["red", "blue"]
    athlete: str
    kata: str
    style: str | None = None
    score: float | None = None
    flag_result: str | None = None
    issue: Literal["win", "loss", "unknown"] = "unknown"
    opponent: str
    opponent_kata: str

    @property
    def global_passage_order(self) -> int:
        """Ordre global du passage dans le live : (match_order - 1) * 2 + passage_order."""
        return (self.match_order - 1) * 2 + self.passage_order
