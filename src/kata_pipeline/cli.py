"""CLI de la pipeline kata."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from kata_pipeline.competition_formats import normalize_competition_type, typed_stage_dir
from kata_pipeline.config import PipelineConfig
from kata_pipeline.storage.cli import drive_app
from kata_pipeline.utils.logging import setup_logging

app = typer.Typer(
    name="kata-pipeline",
    help="Pipeline d'extraction de clips vidéo de kata depuis des lives YouTube.",
    no_args_is_help=True,
)
console = Console()

# Option globale de configuration
CONFIG_OPTION = typer.Option(None, "--config", "-c", help="Chemin vers le fichier de configuration YAML")


def _load_config(config_path: Optional[Path] = None) -> PipelineConfig:
    """Charger et initialiser la configuration."""
    cfg = PipelineConfig.load(config_path)
    cfg.ensure_directories()
    setup_logging(cfg.logging)
    return cfg


def _matches_live(match: object, live: object) -> bool:
    """Vérifie compétition/catégorie/section et le circuit SA/K1."""

    if match.competition != live.competition or match.category != live.category:
        return False
    match_type = normalize_competition_type(match.competition_type, match.competition)
    live_type = normalize_competition_type(live.competition_type, live.competition)
    if match_type is not None and live_type is not None and match_type != live_type:
        return False
    return live.section is None or match.section is None or match.section == live.section


@app.command()
def prepare_video(
    live_id: str = typer.Argument(..., help="Identifiant du live à préparer"),
    lives_file: Path = typer.Option(..., "--lives", "-l", help="Fichier CSV/Excel des lives"),
    config: Optional[Path] = CONFIG_OPTION,
) -> None:
    """Préparer une vidéo : téléchargement optionnel, extraction plage utile, version analyse."""
    cfg = _load_config(config)
    logger = logging.getLogger("kata_pipeline.cli")

    from kata_pipeline.loaders.live_loader import load_lives
    from kata_pipeline.video.preparer import extract_time_range, is_ffmpeg_available

    lives = load_lives(lives_file)
    live = next((lv for lv in lives if lv.id_live == live_id), None)

    if live is None:
        console.print(f"[red]Live '{live_id}' introuvable dans {lives_file}[/red]")
        raise typer.Exit(1)

    # Déterminer la source vidéo
    if live.local_path and live.local_path.exists():
        source = live.local_path
    elif live.url:
        from kata_pipeline.video.downloader import download_video, is_ytdlp_available

        if not is_ytdlp_available():
            console.print("[red]yt-dlp non disponible et pas de vidéo locale.[/red]")
            raise typer.Exit(1)

        # Télécharger vers le local_path partagé (une seule fois par pool) s'il est
        # défini : les tours suivants (T2, T3, PW1) réutiliseront ce fichier au lieu
        # de re-télécharger la vidéo entière. Sinon, fichier propre au live.
        if live.local_path:
            source = live.local_path
        else:
            source = cfg.paths.input_dir / f"{live_id}.mp4"
        console.print(f"Téléchargement: {live.url}")
        source = download_video(live.url, source)
    else:
        console.print("[red]Ni chemin local ni URL disponible pour ce live.[/red]")
        raise typer.Exit(1)

    range_path = cfg.paths.intermediate_dir / f"{live_id}_range.mp4"
    if not range_path.exists():
        extract_time_range(source, range_path, live.useful_start, live.useful_end)
    else:
        logger.info("Range vidéo existante, skip extraction: %s", range_path)

    console.print(f"[green]Vidéo préparée:[/green]")
    console.print(f"  Plage utile: {range_path}")
    console.print(f"  Version analyse: (skip - lecture directe depuis range)")


@app.command()
def compute_motion(
    live_id: str = typer.Argument(..., help="Identifiant du live"),
    config: Optional[Path] = CONFIG_OPTION,
) -> None:
    """Calculer le score de mouvement sur la vidéo d'analyse."""
    cfg = _load_config(config)

    from kata_pipeline.detection.motion_score import compute_motion_scores

    range_path = cfg.paths.intermediate_dir / f"{live_id}_range.mp4"
    if not range_path.exists():
        console.print(f"[red]Vidéo range introuvable: {range_path}[/red]")
        console.print("Lancez d'abord: kata-pipeline prepare-video")
        raise typer.Exit(1)

    console.print(f"Calcul du mouvement: {range_path}")
    result = compute_motion_scores(
        range_path, cfg.motion,
        analysis_fps=cfg.video.analysis_fps,
        analysis_width=cfg.video.analysis_width,
    )

    output_path = cfg.paths.intermediate_dir / f"{live_id}_motion.json"
    result.save(output_path)

    console.print(f"[green]Scores de mouvement sauvegardés: {output_path}[/green]")
    console.print(f"  Frames analysées: {len(result.scores)}")
    console.print(f"  Durée: {result.timestamps[-1]:.1f}s")


@app.command()
def detect_segments(
    live_id: str = typer.Argument(..., help="Identifiant du live"),
    config: Optional[Path] = CONFIG_OPTION,
) -> None:
    """Détecter les segments candidats de kata."""
    cfg = _load_config(config)

    import json

    from kata_pipeline.detection.motion_score import MotionScoreResult
    from kata_pipeline.detection.segment_detector import detect_segments as _detect

    motion_path = cfg.paths.intermediate_dir / f"{live_id}_motion.json"
    if not motion_path.exists():
        console.print(f"[red]Fichier de mouvement introuvable: {motion_path}[/red]")
        console.print("Lancez d'abord: kata-pipeline compute-motion")
        raise typer.Exit(1)

    motion_result = MotionScoreResult.load(motion_path)
    segments = _detect(motion_result, cfg.segments)

    # Sauvegarder les segments
    output_path = cfg.paths.intermediate_dir / f"{live_id}_segments.json"
    segments_data = [
        {
            "start_time": s.start_time,
            "end_time": s.end_time,
            "duration": s.duration,
            "mean_motion": s.mean_motion,
            "max_motion": s.max_motion,
            "confidence": s.confidence,
            "needs_review": s.needs_review,
            "review_reason": s.review_reason,
        }
        for s in segments
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segments_data, f, indent=2)

    console.print(f"[green]Segments détectés: {len(segments)}[/green]")

    table = Table(title="Segments candidats")
    table.add_column("#", style="dim")
    table.add_column("Début")
    table.add_column("Fin")
    table.add_column("Durée")
    table.add_column("Mouvement")
    table.add_column("Confiance")
    table.add_column("Review")

    from kata_pipeline.utils.timecode import seconds_to_timecode

    for i, s in enumerate(segments, 1):
        table.add_row(
            str(i),
            seconds_to_timecode(s.start_time),
            seconds_to_timecode(s.end_time),
            f"{s.duration:.1f}s",
            f"{s.mean_motion:.1f}",
            f"{s.confidence:.2f}",
            "!" if s.needs_review else "",
        )

    console.print(table)


@app.command()
def pair_segments(
    live_id: str = typer.Argument(..., help="Identifiant du live"),
    dataset_file: Path = typer.Option(..., "--dataset", "-d", help="Fichier du dataset sportif"),
    lives_file: Path = typer.Option(..., "--lives", "-l", help="Fichier CSV/Excel des lives"),
    config: Optional[Path] = CONFIG_OPTION,
) -> None:
    """Grouper les segments en paires rouge/bleu."""
    cfg = _load_config(config)

    import json

    from kata_pipeline.detection.pairing import pair_segments as _pair
    from kata_pipeline.detection.segment_detector import CandidateSegment
    from kata_pipeline.loaders.dataset_loader import load_competition_dataset
    from kata_pipeline.loaders.live_loader import find_live_for_match, load_lives

    # Charger les segments
    segments_path = cfg.paths.intermediate_dir / f"{live_id}_segments.json"
    if not segments_path.exists():
        console.print(f"[red]Segments introuvables: {segments_path}[/red]")
        raise typer.Exit(1)

    with open(segments_path, "r", encoding="utf-8") as f:
        segments_data = json.load(f)

    segments = [CandidateSegment(**{k: v for k, v in s.items() if k != "duration"}) for s in segments_data]

    # Charger le dataset pour connaître le nombre de matchs
    matches = load_competition_dataset(dataset_file)
    lives = load_lives(lives_file)
    live = next((lv for lv in lives if lv.id_live == live_id), None)

    if live is None:
        console.print(f"[red]Live '{live_id}' introuvable[/red]")
        raise typer.Exit(1)

    # Filtrer les matchs pour ce live (par competition + category + section si disponible)
    live_matches = [m for m in matches if _matches_live(m, live)]

    # Charger la référence kata
    from kata_pipeline.detection.kata_duration_ref import load_kata_durations

    kata_ref = load_kata_durations(cfg.segments.kata_durations_ref)
    expected_katas: list[tuple[str, str]] | None = None
    if kata_ref is not None:
        expected_katas = [
            (m.model_dump().get("kata_red", ""), m.model_dump().get("kata_blue", ""))
            for m in live_matches
        ]

    result = _pair(
        segments, len(live_matches), cfg.pairing,
        kata_ref=kata_ref, expected_katas=expected_katas,
    )

    # Sauvegarder
    output_path = cfg.paths.intermediate_dir / f"{live_id}_pairs.json"
    pairs_data = {
        "expected_matches": result.expected_matches,
        "detected_matches": result.detected_matches,
        "needs_review": result.needs_review,
        "review_reasons": result.review_reasons,
        "pairs": [
            {
                "match_index": p.match_index,
                "red_start": p.red_segment.start_time,
                "red_end": p.red_segment.end_time,
                "blue_start": p.blue_segment.start_time,
                "blue_end": p.blue_segment.end_time,
                "needs_review": p.needs_review,
                "review_reasons": p.review_reasons,
            }
            for p in result.pairs
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pairs_data, f, indent=2)

    console.print(f"[green]Pairing: {result.detected_matches}/{result.expected_matches} matchs[/green]")
    if result.needs_review:
        console.print(f"[yellow]Attention: {', '.join(result.review_reasons or [])}[/yellow]")


@app.command()
def align(
    live_id: str = typer.Argument(..., help="Identifiant du live"),
    dataset_file: Path = typer.Option(..., "--dataset", "-d", help="Fichier du dataset sportif"),
    lives_file: Path = typer.Option(..., "--lives", "-l", help="Fichier CSV/Excel des lives"),
    config: Optional[Path] = CONFIG_OPTION,
) -> None:
    """Aligner les segments vidéo avec le dataset sportif et générer le dataset de clips."""
    cfg = _load_config(config)

    import json

    import pandas as pd

    from kata_pipeline.alignment.aligner import align_passages_with_segments, build_expected_passages
    from kata_pipeline.detection.pairing import pair_segments as _pair
    from kata_pipeline.detection.segment_detector import CandidateSegment
    from kata_pipeline.loaders.dataset_loader import load_competition_dataset
    from kata_pipeline.loaders.live_loader import load_lives

    # Charger les données
    matches = load_competition_dataset(dataset_file)
    lives = load_lives(lives_file)
    live = next((lv for lv in lives if lv.id_live == live_id), None)

    if live is None:
        console.print(f"[red]Live '{live_id}' introuvable[/red]")
        raise typer.Exit(1)

    live_matches = [m for m in matches if _matches_live(m, live)]

    # Charger segments et faire le pairing
    segments_path = cfg.paths.intermediate_dir / f"{live_id}_segments.json"
    with open(segments_path, "r", encoding="utf-8") as f:
        segments_data = json.load(f)
    segments = [CandidateSegment(**{k: v for k, v in s.items() if k != "duration"}) for s in segments_data]

    # Charger la référence de durées kata si disponible
    from kata_pipeline.detection.kata_duration_ref import load_kata_durations

    kata_ref = load_kata_durations(cfg.segments.kata_durations_ref)

    # Construire la liste des katas attendus par match (pour validation)
    expected_katas: list[tuple[str, str]] | None = None
    if kata_ref is not None:
        match_dicts_for_katas = [m.model_dump() for m in live_matches]
        expected_katas = [
            (m.get("kata_red", ""), m.get("kata_blue", ""))
            for m in match_dicts_for_katas
        ]

    pairing_result = _pair(
        segments, len(live_matches), cfg.pairing,
        kata_ref=kata_ref, expected_katas=expected_katas,
    )

    # Construire les passages attendus
    match_dicts = [m.model_dump() for m in live_matches]
    passages = build_expected_passages(match_dicts, live)

    # Aligner
    clips = align_passages_with_segments(passages, pairing_result, live, kata_ref=kata_ref)

    # Sauvegarder le dataset de clips
    clips_data = [c.model_dump() for c in clips]
    # Convertir Path en str pour JSON
    for clip_d in clips_data:
        for key in ("source_video_path", "clip_path"):
            if clip_d.get(key) is not None:
                clip_d[key] = str(clip_d[key])

    output_path = cfg.paths.output_dir / f"{live_id}_clips.csv"
    df = pd.DataFrame(clips_data)
    df.to_csv(output_path, index=False, encoding="utf-8")

    console.print(f"[green]Dataset de clips généré: {output_path}[/green]")
    console.print(f"  Total clips: {len(clips)}")
    console.print(f"  Auto-validés: {sum(1 for c in clips if c.validation_status == 'auto_validated')}")
    console.print(f"  Needs review: {sum(1 for c in clips if c.needs_review)}")


@app.command()
def generate_clips(
    live_id: str = typer.Argument(..., help="Identifiant du live"),
    config: Optional[Path] = CONFIG_OPTION,
) -> None:
    """Générer les fichiers vidéo de clips à partir du dataset aligné."""
    cfg = _load_config(config)

    import pandas as pd

    from kata_pipeline.schemas.clip import KataClip
    from kata_pipeline.video.clipper import extract_all_clips

    clips_csv = cfg.paths.output_dir / f"{live_id}_clips.csv"
    if not clips_csv.exists():
        console.print(f"[red]Dataset de clips introuvable: {clips_csv}[/red]")
        console.print("Lancez d'abord: kata-pipeline align")
        raise typer.Exit(1)

    df = pd.read_csv(clips_csv)
    clips = [KataClip(**row) for _, row in df.iterrows()]

    # Trouver la vidéo source
    source_path = cfg.paths.intermediate_dir / f"{live_id}_range.mp4"
    if not source_path.exists():
        # Chercher la source originale
        candidates = list(cfg.paths.input_dir.glob(f"{live_id}*"))
        if candidates:
            source_path = candidates[0]
        else:
            console.print(f"[red]Vidéo source introuvable pour {live_id}[/red]")
            raise typer.Exit(1)

    competition_types = {
        normalize_competition_type(clip.competition_type, clip.competition)
        for clip in clips
    }
    competition_types.discard(None)
    if len(competition_types) > 1:
        console.print(
            f"[red]Le live {live_id} mélange plusieurs types de compétition : "
            f"{sorted(competition_types)}[/red]"
        )
        raise typer.Exit(1)
    competition_type = next(iter(competition_types), None)
    output_dir = typed_stage_dir(cfg.paths.clips_pending_dir, competition_type)

    results = extract_all_clips(
        source_path=source_path,
        output_dir=output_dir,
        clips=clips,
        margin_before=cfg.video.clip_margin_before,
        margin_after=cfg.video.clip_margin_after,
    )

    # Conserver dans le CSV le chemin réel, y compris le nouveau niveau SA/K1.
    paths_by_id = {clip.id_clip: str(path) for clip, path in results if path is not None}
    if paths_by_id:
        df["clip_path"] = df.apply(
            lambda row: paths_by_id.get(row["id_clip"], row.get("clip_path")), axis=1
        )
        df.to_csv(clips_csv, index=False, encoding="utf-8")

    extracted = sum(1 for _, p in results if p is not None)
    console.print(
        f"[green]Clips extraits: {extracted}/{len(clips)} -> {output_dir}[/green]"
    )


@app.command()
def run_pipeline(
    live_id: str = typer.Argument(..., help="Identifiant du live"),
    dataset_file: Path = typer.Option(..., "--dataset", "-d", help="Fichier du dataset sportif"),
    lives_file: Path = typer.Option(..., "--lives", "-l", help="Fichier CSV/Excel des lives"),
    config: Optional[Path] = CONFIG_OPTION,
) -> None:
    """Lancer la pipeline complète pour un live."""
    console.print(f"[bold]Pipeline complète pour: {live_id}[/bold]\n")

    console.print("[bold blue]1/5[/bold blue] Préparation vidéo...")
    prepare_video(live_id=live_id, lives_file=lives_file, config=config)

    console.print("\n[bold blue]2/5[/bold blue] Calcul du mouvement...")
    compute_motion(live_id=live_id, config=config)

    console.print("\n[bold blue]3/5[/bold blue] Détection des segments...")
    detect_segments(live_id=live_id, config=config)

    console.print("\n[bold blue]4/5[/bold blue] Pairing et alignement...")
    align(live_id=live_id, dataset_file=dataset_file, lives_file=lives_file, config=config)

    console.print("\n[bold blue]5/5[/bold blue] Génération des clips...")
    generate_clips(live_id=live_id, config=config)

    console.print("\n[bold green]Pipeline terminée avec succès ![/bold green]")


@app.command()
def validate(
    config: Optional[Path] = CONFIG_OPTION,
) -> None:
    """Lancer l'interface de validation Streamlit."""
    import shutil
    import subprocess

    if not shutil.which("streamlit"):
        console.print("[red]Streamlit non installé. Installez avec: pip install streamlit[/red]")
        raise typer.Exit(1)

    app_path = Path(__file__).parent / "validation" / "app.py"
    console.print("Lancement de l'interface de validation...")
    subprocess.run(["streamlit", "run", str(app_path)], check=False)


@app.command("predict-ui")
def predict_ui() -> None:
    """Lancer l'interface Streamlit de prédiction de match."""
    import subprocess
    import sys

    try:
        import streamlit  # noqa: F401
    except ImportError:
        console.print(
            "[red]Streamlit non installé. Installez avec : "
            'pip install -e ".[ui,gnn]"[/red]'
        )
        raise typer.Exit(1) from None

    app_path = Path(__file__).parent / "prediction" / "app.py"
    console.print("Lancement de l'interface de prédiction...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=False)


# Sous-commandes de la phase 2 (poses + GNN), disponibles sous `kata-pipeline gnn ...`.
# L'import est différé pour ne pas exiger torch/mediapipe si l'utilisateur ne s'en sert pas.
try:  # pragma: no cover - dépendances optionnelles
    from kata_pipeline.gnn.cli import gnn_app

    app.add_typer(gnn_app, name="gnn")
except ImportError:  # pragma: no cover
    pass

# Gestion de l'archive Drive, sans charger les bibliothèques Google avant usage.
app.add_typer(drive_app, name="drive")


if __name__ == "__main__":
    app()
