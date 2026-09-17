"""Métriques d'évaluation pour la tâche d'issue (P16).

On mesure la performance de classification (qui gagne ?), la calibration des
probabilités (peut-on faire confiance au "score de confiance" ?) et la
cohérence à l'échange A<->B (le modèle doit prédire l'inverse si on permute).
On fournit aussi une ventilation par sous-groupe (kata, compétition, extracteur,
athlète vu / non vu…).
"""

from __future__ import annotations

import numpy as np

try:
    from sklearn.metrics import (
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    _HAS_SK = True
except ImportError:  # pragma: no cover
    _HAS_SK = False


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """ECE : écart moyen |confiance - exactitude| par bin de confiance."""

    y_true = np.asarray(y_true)
    p = np.asarray(p)
    conf = np.where(p >= 0.5, p, 1.0 - p)
    pred = (p >= 0.5).astype(int)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        ece += mask.sum() / n * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def outcome_metrics(y_true: np.ndarray, p_b: np.ndarray,
                    p_b_swapped: np.ndarray | None = None) -> dict:
    """Calcule toutes les métriques d'issue.

    - ``y_true``      : 1 si B gagne, 0 sinon.
    - ``p_b``         : probabilité prédite que B gagne (ordre A,B).
    - ``p_b_swapped`` : probabilité que B gagne avec l'entrée permutée (B,A) ;
      sert à mesurer la cohérence à l'échange.
    """

    y_true = np.asarray(y_true).astype(int)
    p_b = np.clip(np.asarray(p_b, dtype=float), 1e-7, 1 - 1e-7)
    pred = (p_b >= 0.5).astype(int)

    m: dict[str, float] = {
        "n": int(len(y_true)),
        "accuracy": float((pred == y_true).mean()),
        "brier": float(np.mean((p_b - y_true) ** 2)),
        "ece": expected_calibration_error(y_true, p_b),
    }

    if _HAS_SK:
        both = len(np.unique(y_true)) > 1
        m["balanced_accuracy"] = float(balanced_accuracy_score(y_true, pred))
        m["precision"] = float(precision_score(y_true, pred, zero_division=0))
        m["recall"] = float(recall_score(y_true, pred, zero_division=0))
        m["f1"] = float(f1_score(y_true, pred, zero_division=0))
        m["log_loss"] = float(log_loss(y_true, p_b, labels=[0, 1]))
        m["roc_auc"] = float(roc_auc_score(y_true, p_b)) if both else float("nan")
        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        m["confusion_matrix"] = cm.tolist()
    else:  # pragma: no cover
        m["log_loss"] = float(
            -np.mean(y_true * np.log(p_b) + (1 - y_true) * np.log(1 - p_b))
        )

    if p_b_swapped is not None:
        p_sw = np.clip(np.asarray(p_b_swapped, dtype=float), 1e-7, 1 - 1e-7)
        # Idéal : p_b + p_sw == 1 (antisymétrie). On mesure l'écart moyen…
        m["swap_symmetry_error"] = float(np.mean(np.abs(p_b + p_sw - 1.0)))
        # …et la fraction de prédictions qui s'inversent bien à l'échange.
        pred_sw = (p_sw >= 0.5).astype(int)
        m["swap_consistency"] = float(np.mean(pred_sw != pred))
    return m


def subgroup_metrics(y_true: np.ndarray, p_b: np.ndarray, groups: dict[str, np.ndarray],
                     p_b_swapped: np.ndarray | None = None) -> dict[str, dict]:
    """Métriques d'issue ventilées par valeur de chaque variable de groupe."""

    y_true = np.asarray(y_true)
    p_b = np.asarray(p_b)
    out: dict[str, dict] = {}
    for gname, gvals in groups.items():
        gvals = np.asarray(gvals)
        out[gname] = {}
        for val in np.unique(gvals):
            mask = gvals == val
            if mask.sum() == 0:
                continue
            psw = p_b_swapped[mask] if p_b_swapped is not None else None
            out[gname][str(val)] = outcome_metrics(y_true[mask], p_b[mask], psw)
    return out
