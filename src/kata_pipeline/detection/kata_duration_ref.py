"""Chargement et validation des durées de référence des katas."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Tolérance ajoutée aux bornes min/max par kata (secondes)
TOLERANCE: float = 5.0


@dataclass
class KataDurationBounds:
    """Bornes de durée pour un kata donné."""

    min: float
    max: float
    mean: float
    n: int

    @property
    def safe_min(self) -> float:
        """Min avec tolérance (segment ne devrait pas être plus court).

        Tolérance plus large pour katas avec peu d'échantillons.
        """
        extra = TOLERANCE if self.n >= 3 else TOLERANCE * 2
        return self.min - extra

    @property
    def safe_max(self) -> float:
        """Max avec tolérance (segment ne devrait pas être plus long).

        Tolérance plus large pour katas avec peu d'échantillons.
        """
        extra = TOLERANCE if self.n >= 3 else TOLERANCE * 2
        return self.max + extra


@dataclass
class KataDurationReference:
    """Référence complète des durées de kata."""

    global_min: float
    global_max: float
    global_mean: float
    katas: dict[str, KataDurationBounds]

    def get_bounds(self, kata_name: str) -> KataDurationBounds | None:
        """Obtenir les bornes pour un kata donné.

        Cherche d'abord le nom exact, puis une correspondance partielle.
        """
        if kata_name in self.katas:
            return self.katas[kata_name]

        # Correspondance insensible à la casse
        kata_lower = kata_name.lower()
        for name, bounds in self.katas.items():
            if name.lower() == kata_lower:
                return bounds

        return None

    def is_duration_valid(self, kata_name: str, duration: float) -> bool:
        """Vérifier si une durée est cohérente pour un kata donné."""
        bounds = self.get_bounds(kata_name)
        if bounds is None:
            # Kata inconnu → utiliser les bornes globales
            return (self.global_min - TOLERANCE) <= duration <= (self.global_max + TOLERANCE)
        return bounds.safe_min <= duration <= bounds.safe_max

    def expected_duration(self, kata_name: str) -> float:
        """Durée attendue (moyenne) pour un kata donné."""
        bounds = self.get_bounds(kata_name)
        if bounds is None:
            return self.global_mean
        return bounds.mean

    def get_merge_hint(self, kata_name: str) -> float:
        """Durée max d'une pause à tolérer sans couper, basée sur le kata.

        Les katas longs (Papuren, Suparinpei) ont des pauses plus longues.
        """
        bounds = self.get_bounds(kata_name)
        if bounds is None:
            return 20.0  # défaut

        # Katas longs (>200s mean) → tolérer des pauses jusqu'à 30s
        # Katas courts (<150s mean) → pauses max 15s
        if bounds.mean >= 200:
            return 30.0
        elif bounds.mean >= 160:
            return 25.0
        else:
            return 18.0


def load_kata_durations(ref_path: Path | None = None) -> KataDurationReference | None:
    """Charger le fichier de référence des durées de kata.

    Args:
        ref_path: Chemin vers le fichier YAML de référence

    Returns:
        Référence chargée, ou None si fichier absent/invalide
    """
    if ref_path is None:
        ref_path = Path("data/reference/kata_durations.yaml")

    if not ref_path.exists():
        logger.warning("Fichier de référence kata introuvable: %s", ref_path)
        return None

    with open(ref_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data or "global" not in data or "katas" not in data:
        logger.warning("Format invalide pour %s", ref_path)
        return None

    katas = {}
    for name, info in data["katas"].items():
        katas[name] = KataDurationBounds(
            min=float(info["min"]),
            max=float(info["max"]),
            mean=float(info["mean"]),
            n=int(info["n"]),
        )

    ref = KataDurationReference(
        global_min=float(data["global"]["min"]),
        global_max=float(data["global"]["max"]),
        global_mean=float(data["global"]["mean"]),
        katas=katas,
    )

    logger.info(
        "Référence kata chargée: %d katas, global [%.0f-%.0fs]",
        len(katas), ref.global_min, ref.global_max,
    )
    return ref
