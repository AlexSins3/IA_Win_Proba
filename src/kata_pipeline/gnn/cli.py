"""Sous-commandes CLI de la phase 2 (poses + GNN).

Montées sous ``kata-pipeline gnn ...`` :

    kata-pipeline gnn extract-poses           # tous les clips -> data/poses/*.npz
    kata-pipeline gnn train-comparator        # entraine le juge (vainqueur)
    kata-pipeline gnn train-quality           # entraine la tete de qualite
    kata-pipeline gnn viz-motion CLIP.mp4     # video squelette + vitesse + trainees
    kata-pipeline gnn viz-saliency CLIP.mp4   # video saillance du modele
    kata-pipeline gnn compare A.mp4 B.mp4     # predit le vainqueur entre 2 katas
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from kata_pipeline.competition_formats import normalize_competition_type, typed_competition_dir
from kata_pipeline.gnn.config import load_gnn_config
from kata_pipeline.utils.logging import setup_logging

gnn_app = typer.Typer(
    name="gnn",
    help="Phase 2 : extraction de poses et Graph Neural Network de notation kata.",
    no_args_is_help=True,
)
console = Console()

CONFIG_OPTION = typer.Option(None, "--config", "-c", help="YAML de config phase 2")


def _cfg(path: Optional[Path]):
    setup_logging()
    return load_gnn_config(path)


def _normalize_type_option(value: Optional[str]):
    try:
        return normalize_competition_type(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--competition-type") from exc


@gnn_app.command("extract-poses")
def extract_poses_cmd(
    config: Optional[Path] = CONFIG_OPTION,
    overwrite: bool = typer.Option(False, "--overwrite", help="Recalculer si déjà présent"),
    competition_type: Optional[str] = typer.Option(
        None, "--competition-type", "-t", help="Limiter l'extraction à SA ou K1."
    ),
    include_pending: bool = typer.Option(
        False,
        "--include-pending",
        help="Inclure exceptionnellement les clips encore dans pending.",
    ),
) -> None:
    """Extrait les poses des clips validés liés à un CSV."""

    from kata_pipeline.gnn.data.dataset import build_extraction_manifest
    from kata_pipeline.gnn.pose.base import build_extractor
    from kata_pipeline.gnn.skeletons import get_schema

    cfg = _cfg(config)
    logger = logging.getLogger("kata_pipeline.gnn")
    normalized_type = _normalize_type_option(competition_type)
    manifest = build_extraction_manifest(
        cfg,
        competition_type=normalized_type,
        include_pending=include_pending,
    )
    extractor = build_extractor(cfg.pose, get_schema(cfg.skeleton.schema_name))
    video_store = None

    done = skipped = failed = restored = 0
    for row in manifest.itertuples(index=False):
        pose_path = Path(row.pose_path)
        if pose_path.exists() and not overwrite:
            skipped += 1
            continue
        try:
            clip_path = Path(row.clip_path)
            if not clip_path.exists():
                if not cfg.archive.enabled:
                    raise FileNotFoundError(f"Clip local introuvable : {clip_path}")
                if video_store is None:
                    from kata_pipeline.storage.google_drive import GoogleDriveVideoStore

                    video_store = GoogleDriveVideoStore(cfg.archive)
                clip_path = video_store.resolve_stem(row.stem, row.competition_type)
                restored += 1
            extractor.extract(clip_path).save(pose_path)
            done += 1
        except Exception as exc:  # noqa: BLE001
            logger.error("Echec poses %s : %s", row.clip_path, exc)
            failed += 1
    console.print(
        f"[green]Poses extraites : {done} | restaurées de Drive : {restored} | "
        f"ignorées : {skipped} | échecs : {failed}[/green]"
    )


@gnn_app.command("train-comparator")
def train_comparator_cmd(
    config: Optional[Path] = CONFIG_OPTION,
    competition_type: Optional[str] = typer.Option(
        None, "--competition-type", "-t", help="Entraîner sur SA, K1, ou tous par défaut."
    ),
) -> None:
    """Entraîne le comparateur siamois (prédiction du vainqueur)."""

    from kata_pipeline.gnn.train import train_comparator

    cfg = _cfg(config)
    cfg.train.competition_type = _normalize_type_option(competition_type)
    path = train_comparator(cfg)
    console.print(f"[green]Modèle comparateur : {path}[/green]")


@gnn_app.command("train-quality")
def train_quality_cmd(
    config: Optional[Path] = CONFIG_OPTION,
    competition_type: Optional[str] = typer.Option(
        None, "--competition-type", "-t", help="Entraîner sur SA, K1, ou tous par défaut."
    ),
) -> None:
    """Entraîne la tête de qualité (marge de drapeaux + score)."""

    from kata_pipeline.gnn.train import train_quality

    cfg = _cfg(config)
    cfg.train.competition_type = _normalize_type_option(competition_type)
    path = train_quality(cfg)
    console.print(f"[green]Modèle qualité : {path}[/green]")


@gnn_app.command("train-outcome")
def train_outcome_cmd(
    config: Optional[Path] = CONFIG_OPTION,
    competition_type: Optional[str] = typer.Option(
        None, "--competition-type", "-t", help="Entraîner sur SA, K1, ou tous par défaut."
    ),
) -> None:
    """Entraîne le comparateur d'issue hiérarchique (toutes les fenêtres)."""

    from kata_pipeline.gnn.train_outcome import train_outcome

    cfg = _cfg(config)
    cfg.train.competition_type = _normalize_type_option(competition_type)
    path = train_outcome(cfg)
    console.print(f"[green]Modèle d'issue : {path}[/green]")


@gnn_app.command("predict-match")
def predict_match_cmd(
    video_a: Path = typer.Argument(..., help="Prestation A (.mp4)"),
    video_b: Path = typer.Argument(..., help="Prestation B (.mp4)"),
    model_path: Path = typer.Option(Path("models/outcome.pt"), "--model", "-m"),
    debug: bool = typer.Option(False, "--debug", help="Ajoute des infos de diagnostic"),
    config: Optional[Path] = CONFIG_OPTION,
) -> None:
    """Prédit l'issue entre deux prestations et affiche un JSON (cohérent à l'échange)."""

    import json

    from kata_pipeline.gnn.predict import predict_match

    cfg = _cfg(config)
    result = predict_match(video_a, video_b, model_path, cfg, debug=debug)
    console.print_json(json.dumps(result))


@gnn_app.command("evaluate-outcome")
def evaluate_outcome_cmd(
    model_path: Path = typer.Option(Path("models/outcome.pt"), "--model", "-m"),
    config: Optional[Path] = CONFIG_OPTION,
    competition_type: Optional[str] = typer.Option(
        None, "--competition-type", "-t", help="Évaluer sur SA, K1, ou tous par défaut."
    ),
) -> None:
    """Évalue le modèle d'issue sur le split de validation (métriques + sous-groupes)."""

    import json
    from functools import partial

    from torch.utils.data import DataLoader

    from kata_pipeline.gnn.data.dataset import build_training_manifest, split_manifest
    from kata_pipeline.gnn.data.dataset_hier import (
        PerformancePairDataset,
        collate_pairs,
        compute_in_channels,
    )
    from kata_pipeline.gnn.models.kata_model import KataOutcomeComparator
    from kata_pipeline.gnn.repro import get_device, load_checkpoint
    from kata_pipeline.gnn.skeletons import get_schema
    from kata_pipeline.gnn.train_outcome import _eval_outcome

    cfg = _cfg(config)
    cfg.train.competition_type = _normalize_type_option(competition_type)
    device = get_device()
    schema = get_schema(cfg.skeleton.schema_name)
    in_channels = compute_in_channels(cfg.preprocess)

    manifest = build_training_manifest(cfg)
    _, val_df = split_manifest(manifest, cfg.train.val_ratio, cfg.train.seed, cfg.train.split_by)
    val_ds = PerformancePairDataset(val_df, cfg, train=False)
    if len(val_ds) == 0:
        console.print("[yellow]Aucune paire de validation exploitable.[/yellow]")
        return

    collate = partial(collate_pairs, max_windows=cfg.preprocess.max_windows)
    val_dl = DataLoader(val_ds, batch_size=cfg.train.batch_size, collate_fn=collate)

    model = KataOutcomeComparator(cfg.model, schema, in_channels).to(device)
    load_checkpoint(model_path, model, map_location=str(device))
    metrics = _eval_outcome(model, val_dl, device)
    console.print_json(json.dumps(metrics))



@gnn_app.command("viz-motion")
def viz_motion_cmd(
    clip: Path = typer.Argument(..., help="Clip vidéo .mp4"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Vidéo de sortie"),
    config: Optional[Path] = CONFIG_OPTION,
) -> None:
    """Rend une vidéo squelette + vitesse + trainées (sans modèle)."""

    from kata_pipeline.gnn.viz.overlay import render_motion_overlay

    cfg = _cfg(config)
    out = out or Path(cfg.paths.viz_dir) / f"{clip.stem}_motion.mp4"
    render_motion_overlay(clip, out, cfg)
    console.print(f"[green]Vidéo mouvement : {out}[/green]")


@gnn_app.command("viz-motion-all")
def viz_motion_all_cmd(
    subdir: Optional[str] = typer.Option(
        None,
        "--subdir",
        "-s",
        help="Sous-dossier de clips_dir à traiter (ex. pool_1). Défaut : tous.",
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Regénérer même si la vidéo existe déjà"
    ),
    competition_type: Optional[str] = typer.Option(
        None,
        "--competition-type",
        "-t",
        help="Limiter à SA ou K1 et ranger la sortie sous data/viz/<type>.",
    ),
    config: Optional[Path] = CONFIG_OPTION,
) -> None:
    """Rend l'overlay mouvement pour tous les clips (récursif, barre de progression)."""

    from rich.progress import (
        BarColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    from kata_pipeline.gnn.viz.overlay import render_motion_overlay

    cfg = _cfg(config)
    logger = logging.getLogger("kata_pipeline.gnn")

    normalized_type = _normalize_type_option(competition_type)

    base = Path(cfg.paths.clips_dir)
    if normalized_type:
        base = typed_competition_dir(base, normalized_type)
    if subdir:
        base = base / subdir
    if not base.exists():
        console.print(f"[red]Dossier introuvable : {base}[/red]")
        raise typer.Exit(code=1)

    viz_dir = Path(cfg.paths.viz_dir)
    if normalized_type:
        viz_dir = typed_competition_dir(viz_dir, normalized_type)
    clips = sorted(base.rglob("*.mp4"))
    if not clips:
        console.print(f"[yellow]Aucun clip .mp4 dans {base}[/yellow]")
        raise typer.Exit(code=0)

    done = skipped = failed = 0
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("viz-motion", total=len(clips))
        for clip in clips:
            out = viz_dir / f"{clip.stem}_motion.mp4"
            if out.exists() and not overwrite:
                skipped += 1
                progress.advance(task)
                continue
            progress.update(task, description=f"[cyan]{clip.name[:40]}")
            try:
                render_motion_overlay(clip, out, cfg)
                done += 1
            except Exception as exc:  # noqa: BLE001
                logger.error("Échec viz %s : %s", clip.name, exc)
                failed += 1
            progress.advance(task)

    console.print(
        f"[green]Rendus : {done} | ignorés : {skipped} | échecs : {failed} "
        f"(sortie : {viz_dir})[/green]"
    )

def viz_saliency_cmd(
    clip: Path = typer.Argument(..., help="Clip vidéo .mp4"),
    model_path: Path = typer.Option(
        Path("models/quality.pt"), "--model", "-m", help="Checkpoint du modèle"
    ),
    task: str = typer.Option("quality", "--task", help="quality|comparator"),
    out: Optional[Path] = typer.Option(None, "--out", "-o"),
    config: Optional[Path] = CONFIG_OPTION,
) -> None:
    """Rend une vidéo de saillance : où le modèle regarde pour juger."""

    import torch

    from kata_pipeline.gnn.models.heads import KataComparator, KataQualityNet
    from kata_pipeline.gnn.viz.overlay import render_model_saliency

    cfg = _cfg(config)
    ckpt = torch.load(model_path, map_location="cpu")
    if task == "comparator":
        model = KataComparator(cfg.model)
    else:
        model = KataQualityNet(cfg.model)
    model.load_state_dict(ckpt["model"])

    out = out or Path(cfg.paths.viz_dir) / f"{clip.stem}_saliency.mp4"
    render_model_saliency(clip, out, model, cfg, task=task)
    console.print(f"[green]Vidéo saillance : {out}[/green]")


@gnn_app.command("compare")
def compare_cmd(
    clip_a: Path = typer.Argument(..., help="Kata A (.mp4)"),
    clip_b: Path = typer.Argument(..., help="Kata B (.mp4)"),
    model_path: Path = typer.Option(Path("models/comparator.pt"), "--model", "-m"),
    config: Optional[Path] = CONFIG_OPTION,
) -> None:
    """Prédit le vainqueur entre deux katas (juge anonyme)."""

    import torch

    from kata_pipeline.gnn.models.heads import KataComparator
    from kata_pipeline.gnn.pose.extractor import extract_poses
    from kata_pipeline.gnn.pose.features import compute_node_features, sliding_windows

    cfg = _cfg(config)
    ckpt = torch.load(model_path, map_location="cpu")
    model = KataComparator(cfg.model)
    model.load_state_dict(ckpt["model"])
    model.eval()

    def _to_tensor(clip: Path) -> torch.Tensor:
        data = extract_poses(clip, cfg.pose)
        feats = compute_node_features(
            data["keypoints"], data["visibility"], float(data["fps"]), cfg.graph.normalize
        )
        windows = sliding_windows(feats, cfg.graph.window_size, cfg.graph.window_stride)
        w = windows[len(windows) // 2]
        return torch.from_numpy(w).permute(2, 0, 1).unsqueeze(0).float()

    with torch.no_grad():
        out = model(_to_tensor(clip_a), _to_tensor(clip_b))
        p_b = torch.sigmoid(out["logit"]).item()

    winner = clip_b.name if p_b > 0.5 else clip_a.name
    conf = p_b if p_b > 0.5 else 1 - p_b
    console.print(
        f"[bold]Vainqueur prédit : {winner}[/bold] "
        f"(confiance {conf:.1%} ; qualités latentes "
        f"A={out['quality_a'].item():.3f} B={out['quality_b'].item():.3f})"
    )
