"""Chargement des métadonnées de lives."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from kata_pipeline.competition_formats import CompetitionType, normalize_competition_type
from kata_pipeline.schemas.live import REQUIRED_LIVE_COLUMNS, LiveRecord

logger = logging.getLogger(__name__)


def load_lives(path: Path) -> list[LiveRecord]:
    """Charger la table des lives depuis un fichier CSV ou Excel."""
    logger.info("Chargement des métadonnées de lives: %s", path)

    if not path.exists():
        raise FileNotFoundError(f"Fichier de lives introuvable: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, encoding="utf-8")
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Format non supporté: {suffix}")

    # Vérifier les colonnes requises
    missing = set(REQUIRED_LIVE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans le fichier de lives: {missing}. "
            f"Colonnes requises: {REQUIRED_LIVE_COLUMNS}"
        )

    records: list[LiveRecord] = []
    for _, row in df.iterrows():
        record_data = {}
        for col in df.columns:
            if col in LiveRecord.model_fields:
                val = row.get(col)
                if pd.notna(val):
                    record_data[col] = val

        records.append(LiveRecord(**record_data))

    logger.info("Lives chargés: %d", len(records))
    return records


def find_live_for_match(
    lives: list[LiveRecord],
    competition: str,
    category: str,
    section: str | None = None,
    competition_type: CompetitionType | None = None,
) -> LiveRecord | None:
    """Trouver le live correspondant à une compétition, catégorie et section."""
    for live in lives:
        if live.competition == competition and live.category == category:
            live_type = normalize_competition_type(live.competition_type, live.competition)
            if competition_type is not None and live_type not in (None, competition_type):
                continue
            if section is None or live.section is None or live.section == section:
                return live
    return None
