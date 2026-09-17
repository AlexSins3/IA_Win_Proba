"""Formats de compétition et routage des artefacts SA/K1.

La base historique utilise ``Type_Compet`` tandis que les CSV internes utilisent
``competition_type``. Ce module constitue l'unique source de vérité pour les
tours et les noms de dossiers associés à chaque circuit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

CompetitionType = Literal["SA", "K1"]


@dataclass(frozen=True)
class CompetitionFormat:
    """Tours de poule et phases finales d'un type de compétition."""

    competition_type: CompetitionType
    pool_rounds: tuple[str, ...]
    finals_rounds: tuple[str, ...]
    finals_labels: dict[str, str]

    @property
    def pool_start_round(self) -> str:
        return self.pool_rounds[0]


COMPETITION_FORMATS: dict[CompetitionType, CompetitionFormat] = {
    "SA": CompetitionFormat(
        competition_type="SA",
        pool_rounds=("T1", "T2", "T3", "PW1"),
        finals_rounds=("PW2", "PW3", "Final", "Bronze", "RP1", "RP2", "RP3", "RP4"),
        finals_labels={
            "PW2": "quart",
            "PW3": "demi",
            "Final": "final",
            "Bronze": "bronze",
            "RP1": "rp1",
            "RP2": "rp2",
            "RP3": "rp3",
            "RP4": "rp4",
        },
    ),
    "K1": CompetitionFormat(
        competition_type="K1",
        pool_rounds=("Pool_1", "Pool_2", "Pool_3"),
        finals_rounds=("R1", "R2", "Bronze", "Final"),
        finals_labels={
            "R1": "r1",
            "R2": "r2",
            "Bronze": "bronze",
            "Final": "final",
        },
    ),
}


def normalize_competition_type(
    value: object | None,
    competition: object | None = None,
) -> CompetitionType | None:
    """Normalise ``SA``/``K1`` ou l'infère depuis le nom de compétition.

    Les anciens CSV ne possèdent pas ``competition_type`` mais leurs noms de
    compétition commencent déjà par ``SA_`` ou ``K1_``.
    """

    if value is not None:
        normalized = str(value).strip().upper()
        if normalized in COMPETITION_FORMATS:
            return normalized  # type: ignore[return-value]
        if normalized and normalized not in {"NAN", "NONE"}:
            raise ValueError(f"Type de compétition inconnu : {value!r} (attendu SA ou K1)")

    name = str(competition or "").strip().upper()
    for candidate in COMPETITION_FORMATS:
        if name == candidate or name.startswith(f"{candidate}_"):
            return candidate
    return None


def get_competition_format(value: object) -> CompetitionFormat:
    """Retourne le profil associé à un type de compétition valide."""

    competition_type = normalize_competition_type(value)
    if competition_type is None:
        raise ValueError(f"Impossible de déterminer le format de compétition depuis {value!r}")
    return COMPETITION_FORMATS[competition_type]


def round_slug(round_name: str) -> str:
    """Transforme un nom de tour en suffixe stable pour les sections/lives."""

    slug = re.sub(r"[^a-z0-9]+", "_", round_name.strip().lower()).strip("_")
    return slug or "round"


def typed_competition_dir(base_dir: Path, competition_type: object | None) -> Path:
    """Retourne le dossier d'un circuit sans dupliquer un niveau déjà typé."""

    normalized = normalize_competition_type(competition_type)
    if normalized is None:
        return base_dir
    current_type = base_dir.name.upper()
    if current_type == normalized:
        return base_dir
    if current_type in COMPETITION_FORMATS:
        return base_dir.parent / normalized
    return base_dir / normalized


def typed_stage_dir(stage_dir: Path, competition_type: object | None) -> Path:
    """Insère ``SA`` ou ``K1`` avant un dossier de statut.

    ``data/clips/pending`` devient ``data/clips/SA/pending``. Un chemin déjà
    typé reste inchangé. Sans type (ancien CSV non inférable), le chemin fourni
    est conservé pour compatibilité.
    """

    normalized = normalize_competition_type(competition_type)
    if normalized is None:
        return stage_dir
    return typed_competition_dir(stage_dir.parent, normalized) / stage_dir.name


def competition_type_from_path(path: Path, root: Path) -> CompetitionType | None:
    """Déduit le type depuis le premier segment sous ``root`` si possible."""

    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    if not relative.parts:
        return None
    candidate = relative.parts[0].upper()
    return candidate if candidate in COMPETITION_FORMATS else None  # type: ignore[return-value]
