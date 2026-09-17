"""Configuration du logging pour la pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from kata_pipeline.config import LoggingConfig


def setup_logging(config: LoggingConfig | None = None) -> None:
    """Configurer le logging de la pipeline.

    Args:
        config: Configuration du logging. Si None, utilise les valeurs par défaut.
    """
    if config is None:
        config = LoggingConfig()

    # Créer le dossier de logs
    config.file.parent.mkdir(parents=True, exist_ok=True)

    # Formatter
    formatter = logging.Formatter(config.format)

    # Root logger
    root_logger = logging.getLogger("kata_pipeline")
    root_logger.setLevel(getattr(logging, config.level.upper(), logging.INFO))

    # Handler console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Handler fichier
    file_handler = logging.FileHandler(config.file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
