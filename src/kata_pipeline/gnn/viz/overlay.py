"""Visualisation vidéo des mouvements du corps et de la saillance du modèle.

Deux rendus complémentaires :

1. ``render_motion_overlay`` — dessine le squelette, colore chaque articulation
   selon sa **vitesse** et trace des **trainées** (trails) pour poignets/chevilles.
   Ne nécessite aucun modèle : c'est la visualisation demandée des déplacements
   des parties du corps (épaules, hanches, bras, jambes, mains, pieds...).

2. ``render_model_saliency`` — superpose l'**importance par articulation** telle
   que perçue par un modèle entraîné (gradient de la sortie par rapport aux
   features d'entrée). On voit ainsi *où le modèle regarde* pour juger le kata.

Les deux produisent un fichier ``.mp4`` lisible dans n'importe quel lecteur.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from kata_pipeline.gnn.config import GNNConfig
from kata_pipeline.gnn.graph.skeleton import (
    JOINT_NAMES,
    SKELETON_EDGES,
    SUBSET_LANDMARKS,
)
from kata_pipeline.gnn.pose.features import compute_node_features, sliding_windows

logger = logging.getLogger(__name__)

# Articulations dont on trace la trajectoire (les plus expressives en kata).
TRAIL_JOINTS = ["l_wrist", "r_wrist", "l_ankle", "r_ankle"]
TRAIL_LENGTH = 15


def _speed_to_color(speed: float, vmax: float) -> tuple[int, int, int]:
    """Bleu (lent) -> rouge (rapide), en BGR pour OpenCV."""

    if not np.isfinite(speed):
        speed = 0.0
    t = float(np.clip(speed / (vmax + 1e-6), 0.0, 1.0))
    r = int(255 * t)
    b = int(255 * (1 - t))
    g = int(120 * (1 - abs(0.5 - t) * 2))
    return (b, g, r)


def _extract_pixel_keypoints(
    video_path: str | Path, config: GNNConfig
) -> tuple[np.ndarray, np.ndarray, float, tuple[int, int]]:
    """Re-extrait les poses en coordonnées *pixel* pour l'affichage.

    Retourne (keypoints_px (T,J,2), visibility (T,J), fps, (W,H)).
    """

    from kata_pipeline.gnn.pose.mp_backend import PoseBackend

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, int(round(src_fps / config.pose.target_fps)))

    kps: list[np.ndarray] = []
    viss: list[np.ndarray] = []
    with PoseBackend(config.pose) as backend:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step != 0:
                i += 1
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(i / src_fps * 1000)
            full_kp, full_vis = backend.process(rgb, timestamp_ms)
            i += 1
            kp = np.full((len(SUBSET_LANDMARKS), 2), np.nan, dtype=np.float32)
            vis = np.zeros((len(SUBSET_LANDMARKS),), dtype=np.float32)
            if full_kp is not None:
                for oi, mi in enumerate(SUBSET_LANDMARKS):
                    kp[oi] = (full_kp[mi][0] * w, full_kp[mi][1] * h)
                    vis[oi] = full_vis[mi]
            kps.append(kp)
            viss.append(vis)
    cap.release()
    return np.stack(kps), np.stack(viss), config.pose.target_fps, (w, h)


def _draw_frame(
    canvas: np.ndarray,
    kp_px: np.ndarray,
    speeds: np.ndarray,
    vmax: float,
    trails: dict[str, list[tuple[int, int]]],
    node_importance: np.ndarray | None = None,
) -> None:
    """Dessine squelette, articulations, trainées et (option) importance."""

    # Arêtes.
    for a, b in SKELETON_EDGES:
        pa, pb = kp_px[a], kp_px[b]
        if np.any(np.isnan(pa)) or np.any(np.isnan(pb)):
            continue
        cv2.line(canvas, tuple(pa.astype(int)), tuple(pb.astype(int)), (200, 200, 200), 2)

    # Trainées des extrémités.
    for name, pts in trails.items():
        for k in range(1, len(pts)):
            alpha = k / len(pts)
            color = (int(60 + 195 * alpha), int(60), int(255 - 195 * alpha))
            cv2.line(canvas, pts[k - 1], pts[k], color, max(1, int(1 + 3 * alpha)))

    # Articulations.
    for j, name in enumerate(JOINT_NAMES):
        p = kp_px[j]
        if np.any(np.isnan(p)):
            continue
        base_r = 5
        if node_importance is not None:
            base_r = int(4 + 12 * float(node_importance[j]))
            color = _importance_color(float(node_importance[j]))
        else:
            color = _speed_to_color(float(speeds[j]), vmax)
        cv2.circle(canvas, tuple(p.astype(int)), base_r, color, -1)


def _importance_color(v: float) -> tuple[int, int, int]:
    """Vert (peu) -> jaune -> rouge (important), BGR."""

    v = float(np.clip(v, 0.0, 1.0))
    r = int(255 * min(1.0, 2 * v))
    g = int(255 * min(1.0, 2 * (1 - v)))
    return (0, g, r)


def _open_writer(out_path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """Ouvre un VideoWriter en préférant le H.264 (lisible dans VS Code).

    Essaie ``avc1`` puis ``H264`` (H.264/MP4) ; si le build OpenCV ne les
    fournit pas, retombe sur ``mp4v`` (MPEG-4 Part 2, non lu par le lecteur
    intégré de VS Code mais lisible par VLC / Windows Media Player).
    """

    for tag in ("avc1", "H264"):
        fourcc = cv2.VideoWriter_fourcc(*tag)
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    logger.warning("H.264 indisponible dans OpenCV : repli sur mp4v (ouvrir avec VLC).")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(out_path), fourcc, fps, size)


def render_motion_overlay(
    video_path: str | Path,
    out_path: str | Path,
    config: GNNConfig | None = None,
) -> Path:
    """Rend une vidéo squelette + vitesse + trainées (sans modèle)."""

    config = config or GNNConfig()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kp_px, vis, fps, (w, h) = _extract_pixel_keypoints(video_path, config)
    t = kp_px.shape[0]

    # Vitesses en pixels/frame (pour la couleur).
    vel = np.zeros_like(kp_px)
    vel[1:] = kp_px[1:] - kp_px[:-1]
    speed = np.linalg.norm(vel, axis=2)  # (T, J)
    vmax = float(np.nanpercentile(speed, 95)) if np.isfinite(speed).any() else 1.0
    # Une articulation intermittente (NaN à la frame précédente) produit une
    # vitesse NaN : on la ramène à 0 pour l'affichage.
    speed = np.nan_to_num(speed, nan=0.0)

    writer = _open_writer(out_path, fps, (w, h))
    trails: dict[str, list[tuple[int, int]]] = {n: [] for n in TRAIL_JOINTS}
    idx = {n: JOINT_NAMES.index(n) for n in TRAIL_JOINTS}

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(src_fps / fps)))
    frame_i = 0
    pose_i = 0
    while pose_i < t:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_i % step != 0:
            frame_i += 1
            continue
        frame_i += 1

        for n in TRAIL_JOINTS:
            p = kp_px[pose_i, idx[n]]
            if not np.any(np.isnan(p)):
                trails[n].append(tuple(p.astype(int)))
                trails[n] = trails[n][-TRAIL_LENGTH:]

        _draw_frame(frame, kp_px[pose_i], speed[pose_i], vmax, trails)
        cv2.putText(
            frame,
            "Vitesse: bleu=lent  rouge=rapide",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        writer.write(frame)
        pose_i += 1

    cap.release()
    writer.release()
    logger.info("Vidéo mouvement écrite : %s", out_path)
    return out_path


def compute_joint_saliency(
    model,
    features: np.ndarray,
    config: GNNConfig,
    task: str = "quality",
) -> np.ndarray:
    """Importance par articulation et par frame via gradient d'entrée.

    features : (T, J, C). Retourne (T, J) normalisé dans [0, 1].
    ``task`` : "quality" (tête de qualité) ou "comparator" (qualité latente).
    """

    import torch

    windows = sliding_windows(features, config.graph.window_size, config.graph.window_stride)
    w = windows[len(windows) // 2]  # fenêtre centrale
    x = torch.from_numpy(w).permute(2, 0, 1).unsqueeze(0).float()  # (1,C,T,J)
    x.requires_grad_(True)

    model.eval()
    if task == "comparator":
        out = model.latent_quality(x).sum()
    else:
        out = model(x)["flag"].sum()
    out.backward()

    grad = x.grad.abs().squeeze(0)  # (C, T, J)
    sal = grad.mean(dim=0).detach().cpu().numpy()  # (T, J)
    # Normalisation robuste.
    lo, hi = np.percentile(sal, 5), np.percentile(sal, 95)
    sal = np.clip((sal - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    return sal


def render_model_saliency(
    video_path: str | Path,
    out_path: str | Path,
    model,
    config: GNNConfig | None = None,
    task: str = "quality",
) -> Path:
    """Superpose l'importance par articulation vue par le modèle."""

    config = config or GNNConfig()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kp_px, vis, fps, (w, h) = _extract_pixel_keypoints(video_path, config)
    t = kp_px.shape[0]

    # Reconstruire les features normalisées (mêmes qu'à l'entraînement) pour la saillance.
    kp_norm = np.stack(
        [kp_px[:, :, 0] / w, kp_px[:, :, 1] / h, np.zeros((t, kp_px.shape[1]))], axis=2
    ).astype(np.float32)
    kp_norm = np.nan_to_num(kp_norm)
    feats = compute_node_features(kp_norm, vis, fps, normalize=config.graph.normalize)

    sal_window = compute_joint_saliency(model, feats, config, task=task)  # (Tw, J)

    writer = _open_writer(out_path, fps, (w, h))

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(src_fps / fps)))
    frame_i = 0
    pose_i = 0
    tw = sal_window.shape[0]
    while pose_i < t:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_i % step != 0:
            frame_i += 1
            continue
        frame_i += 1
        # La saillance couvre la fenêtre centrale ; hors fenêtre, on prend la moyenne.
        sal = sal_window[min(pose_i, tw - 1)] if pose_i < tw else sal_window.mean(0)
        _draw_frame(frame, kp_px[pose_i], np.zeros(len(JOINT_NAMES)), 1.0, {}, node_importance=sal)
        cv2.putText(
            frame,
            "Saillance modele: vert=faible  rouge=fort",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        writer.write(frame)
        pose_i += 1

    cap.release()
    writer.release()
    logger.info("Vidéo saillance écrite : %s", out_path)
    return out_path
