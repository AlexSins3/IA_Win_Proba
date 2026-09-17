"""Datasets PyTorch : relie clips vidéo, poses extraites et labels des CSV.

Deux tâches sont couvertes, toutes deux *anonymes* (aucune identité d'athlète) :

1. ``KataWindowDataset`` — une fenêtre de pose -> (marge de drapeaux, score).
   Sert à apprendre une notion de "qualité technique" par kata.

2. ``MatchPairDataset`` — deux katas d'un même match -> vainqueur.
   L'ordre (A, B) est tiré aléatoirement pour éviter tout biais de position ;
   c'est le cœur du "juger n'importe qui" : le modèle compare deux gestes.

Le manifeste d'entraînement lie chaque ligne de CSV directement au ``.npz`` de
poses correspondant. Un manifeste distinct ajoute le clip vidéo uniquement pour
l'étape d'extraction des poses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from kata_pipeline.competition_formats import (
    CompetitionType,
    competition_type_from_path,
    normalize_competition_type,
    typed_competition_dir,
)
from kata_pipeline.gnn.config import GNNConfig
from kata_pipeline.gnn.pose.features import compute_node_features, sliding_windows
from kata_pipeline.utils.filenames import sanitize_filename

logger = logging.getLogger(__name__)

MANIFEST_COLUMNS = [
    "stem",
    "id_match",
    "color",
    "athlete",
    "competition",
    "competition_type",
    "issue",
    "flag_result",
    "score",
    "round",
    "pose_path",
]
EXTRACTION_COLUMNS = [
    *MANIFEST_COLUMNS,
    "clip_path",
    "video_source",
    "drive_file_id",
    "remote_path",
    "remote_md5_checksum",
]


def _expected_stem(row: pd.Series) -> str:
    """Reconstruit le nom de fichier de clip (sans extension) depuis une ligne CSV."""

    try:
        match_order = int(row["match_order"])
    except (ValueError, TypeError):
        match_order = 0
    parts = [
        sanitize_filename(str(row["competition"]), max_length=20),
        sanitize_filename(str(row["category"]), max_length=15),
        sanitize_filename(str(row["round"]), max_length=10),
        f"m{match_order:02d}",
        str(row["color"]),
        sanitize_filename(str(row["athlete"]), max_length=25),
        sanitize_filename(str(row["kata"]), max_length=25),
    ]
    return "_".join(parts)


def _csv_files(output_dir: Path) -> list[Path]:
    csvs = sorted(output_dir.rglob("*_clips.csv"))
    if not csvs:
        raise FileNotFoundError(f"Aucun CSV de clips dans {output_dir}")
    return csvs


def _index_by_stem(paths: list[Path], artifact: str) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in paths:
        previous = index.get(path.stem)
        if previous is None:
            index[path.stem] = path
        elif previous != path:
            duplicates.setdefault(path.stem, [previous]).append(path)
    if duplicates:
        details = "; ".join(
            f"{stem}: {', '.join(map(str, candidates))}"
            for stem, candidates in sorted(duplicates.items())
        )
        raise ValueError(f"Plusieurs {artifact} portent le même stem : {details}")
    return index


def _row_competition_type(row: pd.Series) -> CompetitionType | None:
    return normalize_competition_type(
        row.get("competition_type", row.get("Type_Compet")),
        competition=str(row.get("competition", "")),
    )


def _manifest_row(
    row: pd.Series,
    *,
    stem: str,
    pose_path: Path,
    clip_path: Path | None = None,
) -> dict[str, object]:
    flag = pd.to_numeric(row.get("flag_result"), errors="coerce")
    score = pd.to_numeric(row.get("score"), errors="coerce")
    result: dict[str, object] = {
        "stem": stem,
        "id_match": row["id_match"],
        "color": row["color"],
        "athlete": sanitize_filename(str(row.get("athlete", "")), max_length=25),
        "competition": str(row.get("competition", "")),
        "competition_type": _row_competition_type(row),
        "issue": str(row.get("issue", "")).lower(),
        "flag_result": float(flag) if pd.notna(flag) else np.nan,
        "score": float(score) if pd.notna(score) else np.nan,
        "round": row["round"],
        "pose_path": str(pose_path),
    }
    if clip_path is not None:
        result["clip_path"] = str(clip_path)
    return result


def _finalize_manifest(
    rows: list[dict[str, object]], label: str, *, include_clip_path: bool = False
) -> pd.DataFrame:
    columns = EXTRACTION_COLUMNS if include_clip_path else MANIFEST_COLUMNS
    if not rows:
        logger.info("%s : aucun élément lié à un CSV.", label)
        return pd.DataFrame(columns=columns)
    manifest = (
        pd.DataFrame(rows, columns=columns)
        .drop_duplicates(subset="stem")
        .reset_index(drop=True)
    )
    logger.info("%s : %d éléments liés à un CSV.", label, len(manifest))
    return manifest


def build_training_manifest(
    config: GNNConfig,
    *,
    competition_type: CompetitionType | None = None,
) -> pd.DataFrame:
    """Construit le manifeste d'entraînement depuis les CSV et les poses.

    Les vidéos ne sont ni parcourues ni requises. Cette séparation permet
    d'archiver ``data/clips`` une fois les poses extraites et contrôlées.
    """

    output_dir = Path(config.paths.output_dir)
    poses_dir = Path(config.paths.poses_dir)
    requested_type = normalize_competition_type(
        competition_type if competition_type is not None else config.train.competition_type
    )
    pose_index = _index_by_stem(list(poses_dir.rglob("*.npz")), "fichiers de poses")

    rows: list[dict[str, object]] = []
    for csv in _csv_files(output_dir):
        frame = pd.read_csv(csv)
        for _, row in frame.iterrows():
            if requested_type is not None and _row_competition_type(row) != requested_type:
                continue
            stem = _expected_stem(row)
            pose_path = pose_index.get(stem)
            if pose_path is None:
                continue
            rows.append(_manifest_row(row, stem=stem, pose_path=pose_path))
    return _finalize_manifest(rows, "Manifeste d'entraînement")


def build_extraction_manifest(
    config: GNNConfig,
    *,
    competition_type: CompetitionType | None = None,
    include_pending: bool = False,
) -> pd.DataFrame:
    """Construit le manifeste vidéo nécessaire à l'extraction des poses.

    Les clips validés des arborescences historiques et ``SA``/``K1`` sont
    acceptés. Le dossier ``pending`` et les vidéos dérivées sont exclus par
    défaut. Les nouvelles poses sont rangées sous ``data/poses/SA`` ou
    ``data/poses/K1`` ; une pose historique à la racine reste reconnue.
    """

    output_dir = Path(config.paths.output_dir)
    clips_dir = Path(config.paths.clips_dir)
    poses_dir = Path(config.paths.poses_dir)
    requested_type = normalize_competition_type(competition_type)

    clip_paths: list[Path] = []
    for clip_path in clips_dir.rglob("*.mp4"):
        relative_parts = {part.lower() for part in clip_path.relative_to(clips_dir).parts[:-1]}
        if not include_pending and "pending" in relative_parts:
            continue
        if clip_path.stem.endswith(("_motion", "_saliency")):
            continue
        path_type = competition_type_from_path(clip_path, clips_dir)
        if requested_type is not None and path_type not in (None, requested_type):
            continue
        clip_paths.append(clip_path)

    clip_index = _index_by_stem(clip_paths, "clips")
    pose_index = _index_by_stem(list(poses_dir.rglob("*.npz")), "fichiers de poses")
    remote_index = {}
    remote_store = None
    if config.archive.enabled:
        from kata_pipeline.storage.google_drive import (
            GoogleDriveVideoStore,
            index_by_stem,
            load_archive_index,
        )

        remote_store = GoogleDriveVideoStore(config.archive)
        remote_refs = load_archive_index(config.archive.index_file)
        if not remote_refs:
            logger.warning(
                "Archive Drive activée mais index absent/vide : %s",
                config.archive.index_file,
            )
        remote_index = index_by_stem(
            remote_refs, competition_type=requested_type
        )

    rows: list[dict[str, object]] = []
    for csv in _csv_files(output_dir):
        frame = pd.read_csv(csv)
        for _, row in frame.iterrows():
            row_type = _row_competition_type(row)
            if requested_type is not None and row_type != requested_type:
                continue
            stem = _expected_stem(row)
            clip_path = clip_index.get(stem)
            local_available = clip_path is not None
            remote_ref = remote_index.get(stem)
            if clip_path is None and remote_ref is None:
                continue
            path_type = (
                competition_type_from_path(clip_path, clips_dir)
                if clip_path is not None
                else None
            )
            if path_type is not None and row_type is not None and path_type != row_type:
                logger.warning(
                    "Clip ignoré car son dossier %s contredit le type %s : %s",
                    path_type,
                    row_type,
                    clip_path,
                )
                continue
            if clip_path is None:
                assert remote_store is not None and remote_ref is not None
                clip_path = remote_store.cache_path(remote_ref)
            pose_path = pose_index.get(stem)
            if pose_path is None:
                pose_path = (
                    typed_competition_dir(poses_dir, row_type) / f"{stem}.npz"
                    if row_type
                    else poses_dir / f"{stem}.npz"
                )
            manifest_row = _manifest_row(
                row, stem=stem, pose_path=pose_path, clip_path=clip_path
            )
            manifest_row.update(
                {
                    "video_source": (
                        "local"
                        if local_available
                        else "cache" if clip_path.exists() else "google_drive"
                    ),
                    "drive_file_id": remote_ref.drive_file_id if remote_ref else "",
                    "remote_path": remote_ref.relative_path if remote_ref else "",
                    "remote_md5_checksum": remote_ref.md5_checksum if remote_ref else "",
                }
            )
            rows.append(manifest_row)
    return _finalize_manifest(
        rows, "Manifeste d'extraction", include_clip_path=True
    )


def build_manifest(config: GNNConfig) -> pd.DataFrame:
    """Alias rétrocompatible du manifeste d'entraînement sans vidéos."""

    return build_training_manifest(config)


def load_window(
    pose_path: str | Path,
    config: GNNConfig,
    *,
    train: bool,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Charge un ``.npz`` de poses et retourne une fenêtre (C, T, J).

    - En entraînement : fenêtre tirée aléatoirement (augmentation temporelle).
    - En validation : fenêtre centrale déterministe.
    """

    data = np.load(pose_path, allow_pickle=True)
    visibility = data["visibility"] if "visibility" in data else data["confidence"]
    feats = compute_node_features(
        data["keypoints"],
        visibility,
        float(data["fps"]),
        normalize=config.graph.normalize,
    )  # (T, J, C)
    windows = sliding_windows(feats, config.graph.window_size, config.graph.window_stride)
    if train:
        rng = rng or np.random.default_rng()
        w = windows[rng.integers(len(windows))]
    else:
        w = windows[len(windows) // 2]
    # (T, J, C) -> (C, T, J) pour les convolutions.
    tensor = torch.from_numpy(w).permute(2, 0, 1).contiguous().float()
    return tensor


@dataclass
class _Sample:
    pose_path: str
    flag: float
    score: float


class KataWindowDataset(Dataset):
    """Un kata -> (marge de drapeaux normalisée, score). Cibles NaN masquées."""

    def __init__(self, manifest: pd.DataFrame, config: GNNConfig, train: bool = True):
        self.config = config
        self.train = train
        self.samples = [
            _Sample(r.pose_path, r.flag_result, r.score)
            for r in manifest.itertuples(index=False)
            if Path(r.pose_path).exists()
        ]
        self.rng = np.random.default_rng(config.train.seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        x = load_window(s.pose_path, self.config, train=self.train, rng=self.rng)
        # Normalisation de la marge de drapeaux (0..5 -> 0..1).
        flag = s.flag / float(self.config.model.num_flag_classes - 1)
        return {
            "x": x,
            "flag": torch.tensor(flag if not np.isnan(s.flag) else -1.0, dtype=torch.float32),
            "flag_mask": torch.tensor(0.0 if np.isnan(s.flag) else 1.0),
            "score": torch.tensor(s.score if not np.isnan(s.score) else -1.0, dtype=torch.float32),
            "score_mask": torch.tensor(0.0 if np.isnan(s.score) else 1.0),
        }


class MatchPairDataset(Dataset):
    """Deux katas d'un même match -> vainqueur (0 = A gagne, 1 = B gagne).

    Ordre A/B randomisé à chaque tirage : le modèle ne peut pas exploiter la
    position, il doit comparer les gestes. C'est la tâche cible finale
    ("deux vidéos en entrée -> vainqueur").
    """

    def __init__(self, manifest: pd.DataFrame, config: GNNConfig, train: bool = True):
        self.config = config
        self.train = train
        self.rng = np.random.default_rng(config.train.seed + 1)

        pairs: list[tuple[str, str]] = []  # (winner_pose, loser_pose)
        for _, grp in manifest.groupby("id_match"):
            grp = grp[grp["pose_path"].apply(lambda p: Path(p).exists())]
            if len(grp) != 2:
                continue
            winners = grp[grp["issue"] == "win"]
            losers = grp[grp["issue"] == "loss"]
            if len(winners) == 1 and len(losers) == 1:
                pairs.append(
                    (winners.iloc[0]["pose_path"], losers.iloc[0]["pose_path"])
                )
        self.pairs = pairs
        logger.info("MatchPairDataset : %d paires de match exploitables.", len(pairs))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        winner_path, loser_path = self.pairs[idx]
        xa = load_window(winner_path, self.config, train=self.train, rng=self.rng)
        xb = load_window(loser_path, self.config, train=self.train, rng=self.rng)
        # Tirage de l'ordre : label = index du gagnant.
        if self.train and self.rng.random() < 0.5:
            return {"xa": xb, "xb": xa, "label": torch.tensor(1.0)}
        return {"xa": xa, "xb": xb, "label": torch.tensor(0.0)}


def split_manifest(
    manifest: pd.DataFrame,
    val_ratio: float,
    seed: int,
    split_by: str = "match",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split train/val par *groupe* pour éviter les fuites d'information.

    ``split_by`` :
      - ``"match"``       : les deux katas d'un même combat restent ensemble
        (baseline ; empêche la fuite rouge/bleu intra-match) ;
      - ``"athlete"``     : aucun athlète partagé entre train et val (test
        "athlètes non vus") — un match dont les deux athlètes tombent de part
        et d'autre du split reste simplement inexploitable pour l'appariement ;
      - ``"competition"`` : aucune compétition/source partagée.

    Le regroupement se fait toujours sur une colonne présente ; à défaut on
    retombe sur ``id_match``.
    """

    key_map = {"match": "id_match", "athlete": "athlete", "competition": "competition"}
    key = key_map.get(split_by, "id_match")
    if key not in manifest.columns:
        logger.warning("Colonne '%s' absente ; split par id_match.", key)
        key = "id_match"

    groups = np.array(manifest[key].dropna().unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_val = max(1, int(len(groups) * val_ratio))
    val_groups = set(groups[:n_val])
    val = manifest[manifest[key].isin(val_groups)].reset_index(drop=True)
    train = manifest[~manifest[key].isin(val_groups)].reset_index(drop=True)
    logger.info(
        "Split par '%s' : %d train / %d val (%d groupes, %d en val).",
        key,
        len(train),
        len(val),
        len(groups),
        n_val,
    )
    return train, val
