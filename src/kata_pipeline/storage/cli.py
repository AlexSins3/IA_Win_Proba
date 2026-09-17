"""Commandes de gestion de l'archive vidéo Google Drive."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from kata_pipeline.competition_formats import normalize_competition_type
from kata_pipeline.gnn.config import load_gnn_config
from kata_pipeline.storage.google_drive import GoogleDriveVideoStore

drive_app = typer.Typer(
    name="drive",
    help="Indexation et restauration à la demande des vidéos archivées sur Google Drive.",
    no_args_is_help=True,
)
console = Console()


def _store(config: Optional[Path], folder_id: Optional[str] = None) -> GoogleDriveVideoStore:
    cfg = load_gnn_config(config)
    if folder_id:
        cfg.archive.root_folder_id = folder_id
    return GoogleDriveVideoStore(cfg.archive)


@drive_app.command("auth")
def auth_cmd(
    folder_id: Optional[str] = typer.Option(
        None, "--folder-id", help="ID du dossier Drive `data` (sinon valeur du YAML)."
    ),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Ouvre OAuth dans le navigateur et vérifie l'accès au dossier racine."""

    store = _store(config, folder_id)
    try:
        metadata = store.authenticate()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Authentification Drive impossible : {exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        f"[green]OAuth valide — dossier accessible : {metadata.get('name')} "
        f"({metadata.get('id')})[/green]"
    )


@drive_app.command("index")
def index_cmd(
    folder_id: Optional[str] = typer.Option(
        None, "--folder-id", help="ID du dossier Drive `data` (sinon valeur du YAML)."
    ),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Reconstruit le manifeste local des vidéos présentes sous Drive/data/clips."""

    store = _store(config, folder_id)
    try:
        refs = store.rebuild_index()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Indexation Drive impossible : {exc}[/red]")
        raise typer.Exit(1) from exc
    usable = sum(ref.eligible_for_pose_extraction for ref in refs)
    console.print(
        f"[green]Index Drive : {len(refs)} vidéo(s), {usable} exploitable(s) "
        f"pour les poses -> {store.config.index_file}[/green]"
    )


@drive_app.command("status")
def status_cmd(config: Optional[Path] = typer.Option(None, "--config", "-c")) -> None:
    """Affiche l'état de l'index et du cache, sans appel réseau."""

    store = _store(config)
    refs = store.references()
    by_type: dict[str, int] = {}
    for ref in refs:
        key = ref.competition_type or "inconnu"
        by_type[key] = by_type.get(key, 0) + 1
    console.print(f"Index : {store.config.index_file} — {len(refs)} vidéo(s) {by_type}")
    console.print(
        f"Cache : {store.config.cache_dir} — {store.cache_size_bytes() / 1024**3:.2f} / "
        f"{store.config.cache_max_gb:.2f} Go"
    )


@drive_app.command("fetch")
def fetch_cmd(
    stem: str = typer.Argument(..., help="Nom du clip sans extension."),
    competition_type: Optional[str] = typer.Option(None, "--competition-type", "-t"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Restaure une vidéo précise depuis son identifiant indexé."""

    try:
        normalized_type = normalize_competition_type(competition_type)
        path = _store(config).resolve_stem(stem, normalized_type)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Restauration impossible : {exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[green]Vidéo disponible : {path}[/green]")


@drive_app.command("prune")
def prune_cmd(config: Optional[Path] = typer.Option(None, "--config", "-c")) -> None:
    """Applique immédiatement la limite LRU au cache local."""

    store = _store(config)
    removed = store.prune()
    console.print(
        f"[green]{len(removed)} vidéo(s) évincée(s) ; cache "
        f"{store.cache_size_bytes() / 1024**3:.2f} Go[/green]"
    )
