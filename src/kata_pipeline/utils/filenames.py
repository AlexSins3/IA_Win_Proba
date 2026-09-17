"""Génération de noms de fichiers sûrs pour les clips."""

from __future__ import annotations

import re
import unicodedata

from kata_pipeline.schemas.clip import KataClip


def sanitize_filename(name: str, max_length: int = 50) -> str:
    """Rendre un nom de fichier sûr pour tous les systèmes de fichiers.

    - Normalise les caractères Unicode
    - Remplace les espaces et caractères spéciaux par des underscores
    - Supprime les caractères non-alphanumériques sauf - et _
    - Tronque si nécessaire
    """
    # Normaliser Unicode (décomposer les accents)
    normalized = unicodedata.normalize("NFKD", name)
    # Garder uniquement ASCII
    ascii_str = normalized.encode("ascii", "ignore").decode("ascii")
    # Remplacer espaces et séparateurs par underscore
    cleaned = re.sub(r"[\s\-/\\]+", "_", ascii_str)
    # Supprimer les caractères non-alphanumériques sauf underscore
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "", cleaned)
    # Supprimer les underscores multiples
    cleaned = re.sub(r"_+", "_", cleaned)
    # Supprimer underscores en début/fin
    cleaned = cleaned.strip("_")
    # Tronquer
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip("_")
    # Fallback si vide
    return cleaned or "unnamed"


def generate_clip_filename(clip: KataClip) -> str:
    """Générer un nom de fichier lisible et unique pour un clip.

    Format: {competition}_{category}_{round}_{match_order}_{color}_{athlete}_{kata}.mp4
    """
    parts = [
        sanitize_filename(clip.competition, max_length=20),
        sanitize_filename(clip.category, max_length=15),
        sanitize_filename(clip.round, max_length=10),
        f"m{clip.match_order:02d}",
        clip.color,
        sanitize_filename(clip.athlete, max_length=25),
        sanitize_filename(clip.kata, max_length=25),
    ]

    filename = "_".join(parts) + ".mp4"
    return filename
