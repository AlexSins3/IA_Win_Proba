"""Explicabilité du comparateur d'issue (P9).

Méthodes agnostiques au modèle, par occlusion : on neutralise une partie de
l'entrée et on mesure la variation du logit (score en faveur de B). Cela répond
à « pourquoi ce vainqueur ? » sans dépendre des gradients internes :

- ``joint_importance``   : impact de chaque articulation ;
- ``group_importance``   : impact par groupe (bras/jambes/tronc) ;
- ``window_importance``  : impact de chaque fenêtre temporelle (zones décisives) ;
- ``stability_summary``  : comparaison vitesse/stabilité A vs B (features brutes).
"""

from __future__ import annotations

import numpy as np
import torch

from kata_pipeline.gnn.models.kata_model import KataOutcomeComparator
from kata_pipeline.gnn.skeletons import SkeletonSchema


@torch.no_grad()
def _logit(model: KataOutcomeComparator, xa, mask_a, xb, mask_b) -> float:
    return float(model(xa, mask_a, xb, mask_b)["logit"].item())


@torch.no_grad()
def joint_importance(model: KataOutcomeComparator, xa, mask_a, xb, mask_b,
                     schema: SkeletonSchema) -> dict[str, float]:
    """|Δlogit| quand on neutralise chaque articulation des deux prestations."""

    model.eval()
    base = _logit(model, xa, mask_a, xb, mask_b)
    out: dict[str, float] = {}
    for j, name in enumerate(schema.joint_names):
        a2, b2 = xa.clone(), xb.clone()
        a2[..., j] = 0.0
        b2[..., j] = 0.0
        out[name] = abs(_logit(model, a2, mask_a, b2, mask_b) - base)
    return out


@torch.no_grad()
def group_importance(model: KataOutcomeComparator, xa, mask_a, xb, mask_b,
                     schema: SkeletonSchema) -> dict[str, float]:
    """Importance par groupe anatomique (déduit des noms d'articulations)."""

    model.eval()
    base = _logit(model, xa, mask_a, xb, mask_b)
    groups: dict[str, list[int]] = {"arms": [], "legs": [], "trunk": [], "head": []}
    for j, name in enumerate(schema.joint_names):
        if any(k in name for k in ("shoulder", "elbow", "wrist", "pinky", "index", "thumb")):
            groups["arms"].append(j)
        elif any(k in name for k in ("hip", "knee", "ankle", "heel", "foot")):
            groups["legs"].append(j)
        elif any(k in name for k in ("eye", "ear", "nose", "mouth")):
            groups["head"].append(j)
        else:
            groups["trunk"].append(j)
    out: dict[str, float] = {}
    for g, idxs in groups.items():
        if not idxs:
            continue
        a2, b2 = xa.clone(), xb.clone()
        a2[..., idxs] = 0.0
        b2[..., idxs] = 0.0
        out[g] = abs(_logit(model, a2, mask_a, b2, mask_b) - base)
    return out


@torch.no_grad()
def window_importance(model: KataOutcomeComparator, xa, mask_a, xb, mask_b) -> dict[str, list[float]]:
    """|Δlogit| en masquant chaque fenêtre (zones temporelles décisives)."""

    model.eval()
    base = _logit(model, xa, mask_a, xb, mask_b)

    def per_side(x, mask, other_x, other_mask, side: str):
        vals: list[float] = []
        nw = int(mask.sum().item())
        for w in range(nw):
            m2 = mask.clone()
            m2[..., w] = 0.0
            if side == "a":
                vals.append(abs(_logit(model, x, m2, other_x, other_mask) - base))
            else:
                vals.append(abs(_logit(model, other_x, other_mask, x, m2) - base))
        return vals

    return {
        "a": per_side(xa, mask_a, xb, mask_b, "a"),
        "b": per_side(xb, mask_b, xa, mask_a, "b"),
    }


def stability_summary(xa: torch.Tensor, xb: torch.Tensor,
                      channel_names: list[str]) -> dict[str, dict[str, float]]:
    """Résume vitesse et stabilité de A et B depuis les canaux de mouvement.

    Vitesse = |canaux *vel*| moyen ; stabilité = 1 / (1 + variance des positions).
    """

    def summarize(x: torch.Tensor) -> dict[str, float]:
        arr = x.detach().cpu().numpy()  # (Nw, C, T, J)
        vel_idx = [i for i, n in enumerate(channel_names) if "vel" in n]
        pos_idx = [i for i, n in enumerate(channel_names) if n.startswith("joint_")]
        speed = float(np.abs(arr[:, vel_idx]).mean()) if vel_idx else float("nan")
        var = float(arr[:, pos_idx].var()) if pos_idx else float("nan")
        stability = float(1.0 / (1.0 + var)) if not np.isnan(var) else float("nan")
        return {"mean_speed": speed, "stability": stability}

    return {"a": summarize(xa), "b": summarize(xb)}
