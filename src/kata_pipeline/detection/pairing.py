"""Pairing des segments détectés en matchs rouge/bleu."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from kata_pipeline.config import PairingConfig
from kata_pipeline.detection.kata_duration_ref import KataDurationReference
from kata_pipeline.detection.segment_detector import CandidateSegment

logger = logging.getLogger(__name__)


@dataclass
class MatchPair:
    """Un couple de segments rouge/bleu pour un match."""

    red_segment: CandidateSegment
    blue_segment: CandidateSegment
    match_index: int  # Index du match dans la séquence (1-based)
    needs_review: bool = False
    review_reasons: list[str] | None = None

    def __post_init__(self) -> None:
        if self.review_reasons is None:
            self.review_reasons = []


@dataclass
class PairingResult:
    """Résultat du pairing complet."""

    pairs: list[MatchPair]
    unmatched_segments: list[CandidateSegment]
    expected_matches: int
    detected_matches: int
    needs_review: bool = False
    review_reasons: list[str] | None = None

    def __post_init__(self) -> None:
        if self.review_reasons is None:
            self.review_reasons = []


def pair_segments(
    segments: list[CandidateSegment],
    expected_match_count: int,
    config: PairingConfig | None = None,
    kata_ref: KataDurationReference | None = None,
    expected_katas: list[tuple[str, str]] | None = None,
) -> PairingResult:
    """Grouper les segments candidats en paires rouge/bleu.

    L'ordre est chronologique : rouge puis bleu pour chaque match.

    Args:
        segments: Segments candidats triés chronologiquement
        expected_match_count: Nombre de matchs attendus
        config: Configuration du pairing
        kata_ref: Référence de durées de kata (pour validation)
        expected_katas: Liste de (kata_red, kata_blue) par match, pour validation

    Returns:
        Résultat du pairing avec paires et diagnostics
    """
    if config is None:
        config = PairingConfig()

    # Filtrer les segments valides (durée acceptable)
    valid_segments = [s for s in segments if not s.needs_review or s.duration >= 20.0]
    valid_segments.sort(key=lambda s: s.start_time)

    expected_segments = expected_match_count * 2
    global_review = False
    global_reasons: list[str] = []

    # --- Filtrage intelligent : retirer les segments parasites si sur-détection ---
    if len(valid_segments) > expected_segments:
        excess = len(valid_segments) - expected_segments
        logger.info(
            "Sur-détection: %d segments pour %d attendus, filtrage de %d parasites",
            len(valid_segments), expected_segments, excess,
        )
        # Score de qualité : favorise les segments longs et confiants
        # Les segments courts + needs_review + basse confiance sont les plus parasites
        import math

        def _noise_score(seg: CandidateSegment) -> float:
            """Score bas = plus probable parasite. Segments courts/review/basse conf en premier."""
            duration_factor = math.log(max(seg.duration, 1.0) / 30.0)
            review_penalty = -0.3 if seg.needs_review else 0.0
            return seg.confidence + duration_factor * 0.3 + review_penalty

        scored = [(s, _noise_score(s)) for s in valid_segments]
        scored.sort(key=lambda x: x[1])

        # Retirer les N segments les plus faibles
        to_remove = set(id(s) for s, _ in scored[:excess])
        removed = [s for s in valid_segments if id(s) in to_remove]
        valid_segments = [s for s in valid_segments if id(s) not in to_remove]

        for seg in removed:
            logger.info(
                "  Segment parasite retiré: [%.1fs-%.1fs] dur=%.1fs conf=%.3f",
                seg.start_time, seg.end_time, seg.duration, seg.confidence,
            )

    # Vérification du nombre de segments
    if len(valid_segments) != expected_segments:
        # --- Tentative de merge intelligent ---
        # Si on a plus de segments que prévu, chercher des paires consécutives
        # qui sont trop courtes individuellement mais cohérentes fusionnées
        if len(valid_segments) > expected_segments and kata_ref is not None:
            valid_segments = _try_merge_short_segments(
                valid_segments, expected_segments, kata_ref, expected_katas
            )

    if len(valid_segments) != expected_segments:
        global_review = True
        global_reasons.append(
            f"Nombre de segments ({len(valid_segments)}) != "
            f"attendu ({expected_segments} = {expected_match_count} matchs × 2)"
        )
        logger.warning(
            "Segments détectés: %d, attendus: %d", len(valid_segments), expected_segments
        )

    # Grouper par paires de 2 dans l'ordre chronologique
    pairs: list[MatchPair] = []
    unmatched: list[CandidateSegment] = []

    i = 0
    match_idx = 1

    while i + 1 < len(valid_segments) and match_idx <= expected_match_count:
        red = valid_segments[i]
        blue = valid_segments[i + 1]

        pair_review = False
        pair_reasons: list[str] = []

        # Vérifier le gap entre rouge et bleu
        gap = blue.start_time - red.end_time
        if gap > config.max_gap_within_match:
            pair_review = True
            pair_reasons.append(
                f"Écart rouge-bleu trop grand: {gap:.1f}s > {config.max_gap_within_match}s"
            )

        if gap < 0:
            pair_review = True
            pair_reasons.append(f"Chevauchement rouge-bleu: {gap:.1f}s")

        # Vérifier le ratio de durée
        if red.duration > 0 and blue.duration > 0:
            ratio = min(red.duration, blue.duration) / max(red.duration, blue.duration)
            if ratio < config.duration_ratio_tolerance:
                pair_review = True
                pair_reasons.append(
                    f"Ratio de durée anormal: {ratio:.2f} "
                    f"(rouge={red.duration:.1f}s, bleu={blue.duration:.1f}s)"
                )

        # Vérifier les segments individuels
        if red.needs_review:
            pair_review = True
            pair_reasons.append(f"Segment rouge needs_review: {red.review_reason}")
        if blue.needs_review:
            pair_review = True
            pair_reasons.append(f"Segment bleu needs_review: {blue.review_reason}")

        pair = MatchPair(
            red_segment=red,
            blue_segment=blue,
            match_index=match_idx,
            needs_review=pair_review,
            review_reasons=pair_reasons,
        )
        pairs.append(pair)

        i += 2
        match_idx += 1

    # Segments non appariés
    if i < len(valid_segments):
        unmatched = valid_segments[i:]
        global_review = True
        global_reasons.append(f"{len(unmatched)} segment(s) non apparié(s)")

    # Vérifier les gaps entre matchs successifs
    for j in range(1, len(pairs)):
        prev_blue = pairs[j - 1].blue_segment
        curr_red = pairs[j].red_segment
        inter_match_gap = curr_red.start_time - prev_blue.end_time

        if inter_match_gap < config.min_gap_between_matches:
            pairs[j].needs_review = True
            pairs[j].review_reasons = pairs[j].review_reasons or []
            pairs[j].review_reasons.append(
                f"Écart avec match précédent trop court: {inter_match_gap:.1f}s"
            )

    # --- Validation kata-aware des durées ---
    if kata_ref is not None and expected_katas is not None:
        for idx, pair in enumerate(pairs):
            if idx >= len(expected_katas):
                break
            kata_red, kata_blue = expected_katas[idx]

            # Vérifier durée du segment rouge
            if kata_red and not kata_ref.is_duration_valid(kata_red, pair.red_segment.duration):
                pair.needs_review = True
                pair.review_reasons = pair.review_reasons or []
                bounds = kata_ref.get_bounds(kata_red)
                expected_range = f"[{bounds.safe_min:.0f}-{bounds.safe_max:.0f}s]" if bounds else "?"
                pair.review_reasons.append(
                    f"Durée rouge ({pair.red_segment.duration:.0f}s) hors plage "
                    f"{kata_red} {expected_range}"
                )
                logger.warning(
                    "Match %d: durée rouge %.1fs hors plage pour %s",
                    pair.match_index, pair.red_segment.duration, kata_red,
                )

            # Vérifier durée du segment bleu
            if kata_blue and not kata_ref.is_duration_valid(kata_blue, pair.blue_segment.duration):
                pair.needs_review = True
                pair.review_reasons = pair.review_reasons or []
                bounds = kata_ref.get_bounds(kata_blue)
                expected_range = f"[{bounds.safe_min:.0f}-{bounds.safe_max:.0f}s]" if bounds else "?"
                pair.review_reasons.append(
                    f"Durée bleu ({pair.blue_segment.duration:.0f}s) hors plage "
                    f"{kata_blue} {expected_range}"
                )
                logger.warning(
                    "Match %d: durée bleu %.1fs hors plage pour %s",
                    pair.match_index, pair.blue_segment.duration, kata_blue,
                )

    result = PairingResult(
        pairs=pairs,
        unmatched_segments=unmatched,
        expected_matches=expected_match_count,
        detected_matches=len(pairs),
        needs_review=global_review,
        review_reasons=global_reasons,
    )

    logger.info(
        "Pairing: %d paires créées sur %d attendues (needs_review=%s)",
        len(pairs), expected_match_count, global_review,
    )

    return result


def _try_merge_short_segments(
    segments: list[CandidateSegment],
    expected_count: int,
    kata_ref: KataDurationReference,
    expected_katas: list[tuple[str, str]] | None,
) -> list[CandidateSegment]:
    """Fusionner des segments consécutifs trop courts si leur combinaison est cohérente.

    Cas typique : un kata avec une pause (Papuren) est détecté en 2 morceaux.
    Si les 2 morceaux sont trop courts individuellement mais leur somme est dans
    la plage attendue, on les fusionne.

    Args:
        segments: Segments triés chronologiquement
        expected_count: Nombre de segments attendus
        kata_ref: Référence de durées
        expected_katas: Katas attendus (red, blue) par match

    Returns:
        Liste potentiellement réduite de segments après merges
    """
    if not expected_katas:
        return segments

    # Seuil de gap max entre deux segments pour être éligibles au merge
    max_merge_gap = 60.0  # secondes

    # Durée min en dessous de laquelle un segment est "trop court"
    global_safe_min = kata_ref.global_min - 5.0

    # Construire la liste plate des katas attendus (red1, blue1, red2, blue2, ...)
    flat_katas = []
    for kata_red, kata_blue in expected_katas:
        flat_katas.append(kata_red)
        flat_katas.append(kata_blue)

    merged = list(segments)
    changed = True

    while changed and len(merged) > expected_count:
        changed = False
        for i in range(len(merged) - 1):
            seg_a = merged[i]
            seg_b = merged[i + 1]

            # Les deux segments sont-ils courts ?
            both_short = seg_a.duration < global_safe_min and seg_b.duration < global_safe_min

            # Un seul est court mais le gap est petit ?
            one_short = (seg_a.duration < global_safe_min or seg_b.duration < global_safe_min)
            gap = seg_b.start_time - seg_a.end_time

            if gap > max_merge_gap:
                continue

            if not (both_short or (one_short and gap < 30.0)):
                continue

            # Durée combinée
            combined_duration = seg_b.end_time - seg_a.start_time

            # Vérifier si la durée combinée correspond à un kata attendu
            # à cette position dans la séquence
            valid_combined = False
            if i < len(flat_katas):
                kata_name = flat_katas[i]
                if kata_name and kata_ref.is_duration_valid(kata_name, combined_duration):
                    valid_combined = True

            # Ou si la combinaison est au moins dans les bornes globales
            if not valid_combined and kata_ref.global_min - 5 <= combined_duration <= kata_ref.global_max + 5:
                valid_combined = True

            if valid_combined:
                # Fusionner les deux segments
                combined_scores = [seg_a.mean_motion, seg_b.mean_motion]
                new_seg = CandidateSegment(
                    start_time=seg_a.start_time,
                    end_time=seg_b.end_time,
                    mean_motion=sum(combined_scores) / len(combined_scores),
                    max_motion=max(seg_a.max_motion, seg_b.max_motion),
                    confidence=min(seg_a.confidence, seg_b.confidence),
                    needs_review=False,
                    review_reason=None,
                )
                logger.info(
                    "  Merge: [%.1fs-%.1fs](%.1fs) + [%.1fs-%.1fs](%.1fs) → %.1fs (gap=%.1fs)",
                    seg_a.start_time, seg_a.end_time, seg_a.duration,
                    seg_b.start_time, seg_b.end_time, seg_b.duration,
                    new_seg.duration, gap,
                )
                merged = merged[:i] + [new_seg] + merged[i + 2:]
                changed = True
                break

    if len(merged) != len(segments):
        logger.info(
            "Merge intelligent: %d → %d segments (attendus: %d)",
            len(segments), len(merged), expected_count,
        )

    return merged
