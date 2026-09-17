"""Chargement du dataset de compétition."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from kata_pipeline.schemas.competition import REQUIRED_COLUMNS, MatchRecord

logger = logging.getLogger(__name__)


def load_competition_dataset(path: Path) -> list[MatchRecord]:
    """Charger le dataset de compétition depuis un fichier CSV ou Excel.

    Vérifie la présence des colonnes requises et retourne une liste de MatchRecord.
    """
    logger.info("Chargement du dataset de compétition: %s", path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, encoding="utf-8")
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Format non supporté: {suffix}. Utiliser .csv, .xlsx ou .parquet")

    # Vérifier les colonnes requises
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans le dataset: {missing}. "
            f"Colonnes requises: {REQUIRED_COLUMNS}"
        )

    # Trier par ordre de match
    df = df.sort_values(["competition", "category", "round", "match_order"]).reset_index(drop=True)

    # Convertir en MatchRecord
    records: list[MatchRecord] = []
    for _, row in df.iterrows():
        record_data = {col: row.get(col) for col in REQUIRED_COLUMNS}
        # Ajouter les colonnes optionnelles présentes
        for col in df.columns:
            if col not in REQUIRED_COLUMNS and col in MatchRecord.model_fields:
                val = row.get(col)
                if pd.notna(val):
                    record_data[col] = val

        records.append(MatchRecord(**record_data))

    logger.info("Dataset chargé: %d matchs", len(records))
    return records


def load_competition_dataframe(path: Path) -> pd.DataFrame:
    """Charger le dataset brut comme DataFrame pour inspection."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8")
    elif suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    elif suffix == ".parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Format non supporté: {suffix}")
