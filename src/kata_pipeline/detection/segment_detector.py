"""Détection de segments candidats à partir de la courbe de mouvement."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from kata_pipeline.config import SegmentsConfig
from kata_pipeline.detection.motion_score import MotionScoreResult

logger = logging.getLogger(__name__)


@dataclass
class CandidateSegment:
    """Un segment candidat détecté comme probable kata."""

    start_time: float
    end_time: float
    mean_motion: float
    max_motion: float
    confidence: float
    needs_review: bool = False
    review_reason: str | None = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


def detect_segments(
    motion_result: MotionScoreResult,
    config: SegmentsConfig | None = None,
) -> list[CandidateSegment]:
    """Détecter les segments de kata candidats depuis la courbe de mouvement.

    Stratégie hybride :
    1. Détecter les pics de transition (score > transition_threshold pendant > 3s)
       → Ces pics correspondent aux replays, changements de caméra, overlays.
    2. Identifier les "vallées" entre pics = zones candidates contenant les katas.
    3. Au sein de chaque vallée, trouver la sous-zone avec le plus d'activité
       soutenue (>= min_motion_score) pour localiser le kata.
    4. Filtrer par durée min/max.

    Args:
        motion_result: Résultat du calcul de mouvement
        config: Configuration des paramètres de détection

    Returns:
        Liste de segments candidats triés chronologiquement
    """
    if config is None:
        config = SegmentsConfig()

    scores = motion_result.smoothed_scores
    timestamps = motion_result.timestamps
    fps = motion_result.fps

    if not scores or len(scores) < 10:
        return []

    duration = timestamps[-1]

    # --- Phase 1 : Détecter les transitions (pics hauts) ---
    transition_threshold = max(config.min_motion_score * 2, 1.0)
    min_transition_frames = max(2, int(fps * 3))  # au moins 3 secondes

    transitions = _find_transitions(scores, timestamps, fps, transition_threshold, min_transition_frames)
    logger.info("Transitions détectées: %d (seuil=%.2f)", len(transitions), transition_threshold)

    # --- Phase 2 : Définir les vallées entre transitions ---
    valleys = _find_valleys(transitions, duration, min_valley_duration=config.min_duration * 0.5)
    logger.info("Vallées candidates: %d", len(valleys))

    if not valleys:
        # Fallback : pas de transition claire, traiter toute la section
        logger.warning("Aucune transition détectée, fallback sur la section entière")
        valleys = [(0.0, duration)]

    # --- Phase 3 : Localiser le kata dans chaque vallée ---
    # Pour les vallées larges (> max_duration * 1.3), tenter de les scinder
    # car elles contiennent probablement plusieurs katas
    segments: list[CandidateSegment] = []
    for valley_start, valley_end in valleys:
        valley_dur = valley_end - valley_start
        if valley_dur > config.max_duration * 1.3:
            # Vallée trop large → scinder en sous-vallées via le creux de mouvement
            sub_valleys = _split_wide_valley(
                scores, timestamps, fps, valley_start, valley_end, config
            )
            for sv_start, sv_end in sub_valleys:
                seg = _find_kata_in_valley(
                    scores, timestamps, fps, sv_start, sv_end, config
                )
                if seg is not None:
                    segments.append(seg)
        else:
            segment = _find_kata_in_valley(
                scores, timestamps, fps, valley_start, valley_end, config
            )
            if segment is not None:
                segments.append(segment)

    logger.info(
        "Segments candidats: %d (dont %d needs_review)",
        len(segments),
        sum(1 for s in segments if s.needs_review),
    )

    return segments


def _find_transitions(
    scores: list[float],
    timestamps: list[float],
    fps: float,
    threshold: float,
    min_frames: int,
) -> list[tuple[float, float]]:
    """Trouver les zones de transition (pics de mouvement élevé)."""
    transitions: list[tuple[float, float]] = []
    in_transition = False
    trans_start = 0

    for i, score in enumerate(scores):
        if score >= threshold:
            if not in_transition:
                in_transition = True
                trans_start = i
        else:
            if in_transition:
                if (i - trans_start) >= min_frames:
                    t_start = timestamps[trans_start] if trans_start < len(timestamps) else trans_start / fps
                    t_end = timestamps[i - 1] if (i - 1) < len(timestamps) else (i - 1) / fps
                    transitions.append((t_start, t_end))
                in_transition = False

    # Clôturer si on finit dans une transition
    if in_transition and (len(scores) - trans_start) >= min_frames:
        t_start = timestamps[trans_start] if trans_start < len(timestamps) else trans_start / fps
        t_end = timestamps[-1]
        transitions.append((t_start, t_end))

    return transitions


def _find_valleys(
    transitions: list[tuple[float, float]],
    total_duration: float,
    min_valley_duration: float,
) -> list[tuple[float, float]]:
    """Trouver les vallées (espaces entre transitions)."""
    valleys: list[tuple[float, float]] = []

    if not transitions:
        return [(0.0, total_duration)]

    # Avant la première transition
    if transitions[0][0] > min_valley_duration:
        valleys.append((0.0, transitions[0][0]))

    # Entre les transitions
    for i in range(len(transitions) - 1):
        gap_start = transitions[i][1]
        gap_end = transitions[i + 1][0]
        if (gap_end - gap_start) >= min_valley_duration:
            valleys.append((gap_start, gap_end))

    # Après la dernière transition
    last_end = transitions[-1][1]
    if (total_duration - last_end) > min_valley_duration:
        valleys.append((last_end, total_duration))

    return valleys


def _split_wide_valley(
    scores: list[float],
    timestamps: list[float],
    fps: float,
    valley_start: float,
    valley_end: float,
    config: SegmentsConfig,
) -> list[tuple[float, float]]:
    """Scinder une vallée large en sous-vallées au point le plus calme.

    Utilise une fenêtre glissante pour trouver la zone avec le moins de mouvement
    (typiquement le changement d'athlète entre deux katas).

    Amélioration : ne scinde que si le creux est vraiment un inter-kata (très bas)
    et non une simple pause au sein d'un kata (Papuren, Suparinpei, etc.).
    Critères :
    - Le creux doit être en dessous de 30% de la moyenne de la vallée
    - Chaque sous-vallée doit être >= min_duration
    """
    valley_dur = valley_end - valley_start

    # Extraire les scores de la vallée
    valley_indices = [
        i for i, t in enumerate(timestamps)
        if valley_start <= t <= valley_end
    ]
    if len(valley_indices) < 10:
        return [(valley_start, valley_end)]

    valley_scores = [scores[i] for i in valley_indices]
    valley_ts = [timestamps[i] for i in valley_indices]

    # Moyenne de la vallée pour référence
    valley_mean = sum(valley_scores) / len(valley_scores)

    # Fenêtre glissante de 15s pour trouver le creux
    window_size = max(3, int(fps * 15))
    min_score = float("inf")
    min_center_idx = len(valley_scores) // 2

    # Ne pas chercher trop près des bords (au moins min_duration de chaque côté)
    margin = max(int(len(valley_scores) * 0.2), int(fps * config.min_duration))
    search_start = margin
    search_end = len(valley_scores) - margin

    if search_start >= search_end:
        return [(valley_start, valley_end)]

    for i in range(search_start, search_end - window_size + 1):
        window = valley_scores[i:i + window_size]
        avg = sum(window) / len(window)
        if avg < min_score:
            min_score = avg
            min_center_idx = i + window_size // 2

    # GARDE-FOU : ne pas scinder si le creux n'est pas significativement bas
    # Un vrai inter-kata (athlètes qui changent) a un mouvement très faible
    # Une pause intra-kata (Papuren, etc.) garde un niveau de mouvement modéré
    split_threshold = valley_mean * 0.30
    if min_score > split_threshold:
        logger.debug(
            "Vallée [%.1f-%.1f] non scindée: creux=%.2f > seuil=%.2f (30%% de mean=%.2f)",
            valley_start, valley_end, min_score, split_threshold, valley_mean,
        )
        return [(valley_start, valley_end)]

    # Scinder au centre du creux
    split_time = valley_ts[min_center_idx]

    sub_valleys = []
    # Première sous-vallée : vérifier durée minimale
    if (split_time - valley_start) >= config.min_duration:
        sub_valleys.append((valley_start, split_time))
    # Seconde sous-vallée : vérifier durée minimale
    if (valley_end - split_time) >= config.min_duration:
        sub_valleys.append((split_time, valley_end))

    if not sub_valleys:
        # Les sous-vallées seraient trop courtes → ne pas scinder
        return [(valley_start, valley_end)]

    # Si on n'a qu'un morceau valide, garder la vallée entière
    if len(sub_valleys) == 1:
        return [(valley_start, valley_end)]

    # Récursivement scinder si encore trop large
    result = []
    for sv_start, sv_end in sub_valleys:
        sv_dur = sv_end - sv_start
        if sv_dur > config.max_duration * 1.3:
            result.extend(_split_wide_valley(
                scores, timestamps, fps, sv_start, sv_end, config
            ))
        else:
            result.append((sv_start, sv_end))

    return result


def _find_kata_in_valley(
    scores: list[float],
    timestamps: list[float],
    fps: float,
    valley_start: float,
    valley_end: float,
    config: SegmentsConfig,
) -> CandidateSegment | None:
    """Localiser le segment de kata au sein d'une vallée.

    Stratégie révisée :
    - Les transitions (pics hauts) délimitent déjà bien les vallées.
    - Chaque vallée contient typiquement UN kata (une performance).
    - Pour les katas longs et contrôlés (Chatanyara, Papuren, Anan Dai),
      le mouvement peut être très faible → on ne peut PAS se fier au seuil.
    - Approche : utiliser la vallée presque entière, trimée aux bords,
      puis affiner si possible avec un seuil très bas.
    """
    # Extraire les scores de la vallée
    valley_indices = [
        i for i, t in enumerate(timestamps)
        if valley_start <= t <= valley_end
    ]

    if not valley_indices:
        return None

    valley_scores = [scores[i] for i in valley_indices]
    valley_ts = [timestamps[i] for i in valley_indices]
    valley_duration = valley_end - valley_start

    # Calculer la moyenne globale de la vallée
    mean_valley = sum(valley_scores) / len(valley_scores) if valley_scores else 0.0
    max_valley = max(valley_scores) if valley_scores else 0.0

    # --- Approche principale : utiliser un seuil TRÈS bas pour trouver les bornes ---
    # Le seuil est juste au-dessus du bruit de fond (caméra statique ≈ 0-0.5)
    activity_threshold = max(0.5, mean_valley * 0.15)

    # Trouver les régions actives en tolérant les pauses courtes (merge_gap)
    # Cela empêche de couper un kata lors d'une pause (ex: Papuren)
    merge_gap_frames = int(fps * config.merge_gap)

    first_active = None
    last_active = None
    gap_since_last = 0

    for idx, score in enumerate(valley_scores):
        if score >= activity_threshold:
            if first_active is None:
                first_active = idx
            last_active = idx
            gap_since_last = 0
        else:
            if first_active is not None:
                gap_since_last += 1
                # Si le gap est encore dans la tolérance, on continue
                # (on ne "coupe" pas le segment)
                if gap_since_last <= merge_gap_frames:
                    last_active = idx

    if first_active is None or last_active is None:
        # Aucune activité détectée — utiliser la vallée trimée si durée OK
        if config.min_duration <= valley_duration <= config.max_duration:
            trim_start = valley_start + min(8.0, valley_duration * 0.05)
            trim_end = valley_end - min(10.0, valley_duration * 0.07)
            return CandidateSegment(
                start_time=trim_start,
                end_time=trim_end,
                mean_motion=mean_valley,
                max_motion=max_valley,
                confidence=0.35,
                needs_review=True,
                review_reason="Pas de signal, vallée trimée",
            )
        return None

    # Bornes basées sur la première/dernière activité
    seg_start_time = valley_ts[first_active]
    seg_end_time = valley_ts[last_active]
    seg_duration = seg_end_time - seg_start_time

    # Ajouter une petite marge avant/après (salutation, prise de position)
    margin_before = min(5.0, seg_start_time - valley_start)
    margin_after = min(5.0, valley_end - seg_end_time)
    seg_start_time = max(valley_start, seg_start_time - margin_before)
    seg_end_time = min(valley_end, seg_end_time + margin_after)
    seg_duration = seg_end_time - seg_start_time

    # Calculer les stats sur la région détectée
    region_scores = valley_scores[first_active:last_active + 1]
    mean_motion = sum(region_scores) / len(region_scores) if region_scores else 0.0
    max_motion = max(region_scores) if region_scores else 0.0

    # Vérifier la durée
    needs_review = False
    review_reason = None
    if seg_duration < config.min_duration:
        # Trop court — étendre aux bornes de la vallée avec trim conservateur
        seg_start_time = valley_start + min(5.0, valley_duration * 0.03)
        seg_end_time = valley_end - min(8.0, valley_duration * 0.05)
        seg_duration = seg_end_time - seg_start_time
        needs_review = True
        review_reason = "Segment étendu (original trop court)"
    elif seg_duration > config.max_duration:
        needs_review = True
        review_reason = f"Durée longue: {seg_duration:.1f}s > {config.max_duration}s"

    confidence = _compute_confidence(seg_duration, mean_motion, config)

    return CandidateSegment(
        start_time=seg_start_time,
        end_time=seg_end_time,
        mean_motion=mean_motion,
        max_motion=max_motion,
        confidence=confidence,
        needs_review=needs_review,
        review_reason=review_reason,
    )


def _compute_confidence(
    duration: float, mean_motion: float, config: SegmentsConfig
) -> float:
    """Calculer un score de confiance entre 0 et 1.

    Facteurs :
    - Durée dans la plage typique d'un kata (global_mean ± écart)
    - Intensité de mouvement supérieure au seuil
    """
    # Score de durée (gaussien centré sur la moyenne globale des katas)
    ideal_duration = 187.0  # moyenne observée sur 76 clips validés
    duration_score = max(0.0, 1.0 - abs(duration - ideal_duration) / ideal_duration)

    # Score de mouvement (normalisé par rapport au seuil)
    if mean_motion <= 0:
        motion_score = 0.0
    else:
        motion_score = min(1.0, mean_motion / (config.min_motion_score * 3))

    # Dans les bornes min/max ?
    in_range = config.min_duration <= duration <= config.max_duration
    range_bonus = 0.2 if in_range else 0.0

    confidence = (duration_score * 0.4 + motion_score * 0.4 + range_bonus) 
    return round(min(1.0, confidence), 3)
