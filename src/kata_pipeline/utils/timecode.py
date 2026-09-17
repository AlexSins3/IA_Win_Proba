"""Conversion de timecodes."""

from __future__ import annotations


def timecode_to_seconds(timecode: str) -> float:
    """Convertir un timecode HH:MM:SS ou MM:SS en secondes.

    Supporte aussi HH:MM:SS.mmm pour les millisecondes.

    Args:
        timecode: Chaîne au format HH:MM:SS, MM:SS ou HH:MM:SS.mmm

    Returns:
        Nombre de secondes

    Raises:
        ValueError: Si le format est invalide
    """
    parts = timecode.strip().split(":")
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    else:
        raise ValueError(f"Format de timecode invalide: '{timecode}'. Attendu HH:MM:SS ou MM:SS")


def seconds_to_timecode(seconds: float, include_ms: bool = False) -> str:
    """Convertir des secondes en timecode HH:MM:SS.

    Args:
        seconds: Nombre de secondes
        include_ms: Inclure les millisecondes

    Returns:
        Chaîne au format HH:MM:SS ou HH:MM:SS.mmm
    """
    if seconds < 0:
        raise ValueError(f"Nombre de secondes négatif: {seconds}")

    hours = int(seconds // 3600)
    remaining = seconds - hours * 3600
    minutes = int(remaining // 60)
    secs = remaining - minutes * 60

    if include_ms:
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    else:
        return f"{hours:02d}:{minutes:02d}:{int(secs):02d}"
