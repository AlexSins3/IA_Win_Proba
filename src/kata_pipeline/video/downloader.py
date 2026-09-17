"""Téléchargement optionnel de vidéos YouTube via yt-dlp."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def is_ytdlp_available() -> bool:
    """Vérifier si yt-dlp est installé et disponible."""
    return shutil.which("yt-dlp") is not None


def download_video(
    url: str,
    output_path: Path,
    quality: str = "best",
    start_time: float | None = None,
    end_time: float | None = None,
    cookies_from_browser: str | None = None,
    cookies_file: Path | str | None = None,
) -> Path:
    """Télécharger une vidéo YouTube via yt-dlp.

    Args:
        url: URL YouTube
        output_path: Chemin de sortie pour la vidéo
        quality: Qualité souhaitée (best, medium, low)
        start_time: Début de la plage à télécharger (secondes)
        end_time: Fin de la plage à télécharger (secondes)
        cookies_from_browser: Navigateur d'où extraire les cookies
            (ex. "chrome", "firefox", "edge"). À défaut, la variable
            d'environnement KATA_YTDLP_COOKIES_BROWSER est utilisée.
        cookies_file: Chemin d'un fichier cookies.txt. À défaut, la
            variable d'environnement KATA_YTDLP_COOKIES_FILE est utilisée.

    Returns:
        Chemin du fichier téléchargé
    """
    if not is_ytdlp_available():
        raise RuntimeError(
            "yt-dlp n'est pas installé. Installez-le avec: pip install yt-dlp\n"
            "Ou placez la vidéo manuellement dans le dossier d'entrée."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Construire la commande yt-dlp
    format_map = {
        "best": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "medium": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]",
        "low": "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480]",
    }

    cmd = [
        "yt-dlp",
        "-f", format_map.get(quality, format_map["best"]),
        "-o", str(output_path),
        "--no-playlist",
        "--retries", "infinite",
        "--fragment-retries", "infinite",
        "--http-chunk-size", "10M",
    ]

    # Authentification via cookies (contourne "Sign in to confirm you're not a bot").
    # Priorité: argument explicite > variable d'environnement.
    browser = cookies_from_browser or os.environ.get("KATA_YTDLP_COOKIES_BROWSER")
    cookies_txt = cookies_file or os.environ.get("KATA_YTDLP_COOKIES_FILE")
    if browser:
        cmd.extend(["--cookies-from-browser", browser])
        logger.info("Utilisation des cookies du navigateur: %s", browser)
    elif cookies_txt:
        cmd.extend(["--cookies", str(cookies_txt)])
        logger.info("Utilisation du fichier cookies: %s", cookies_txt)

    # yt-dlp supporte les sections de téléchargement
    if start_time is not None or end_time is not None:
        section = "*"
        if start_time is not None:
            section += f"{start_time}"
        section += "-"
        if end_time is not None:
            section += f"{end_time}"
        cmd.extend(["--download-sections", section])

    cmd.append(url)

    logger.info("Téléchargement: %s -> %s", url, output_path)
    logger.debug("Commande: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        logger.error("Erreur yt-dlp: %s", result.stderr)
        raise RuntimeError(f"Échec du téléchargement: {result.stderr}")

    # yt-dlp peut ajouter une extension, chercher le fichier
    if output_path.exists():
        return output_path

    # Chercher le fichier avec extension ajoutée
    for candidate in output_path.parent.glob(f"{output_path.stem}*"):
        if candidate.is_file():
            logger.info("Fichier téléchargé: %s", candidate)
            return candidate

    raise FileNotFoundError(f"Fichier téléchargé introuvable après yt-dlp: {output_path}")
