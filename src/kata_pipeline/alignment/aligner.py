"""Alignement des segments détectés avec le dataset sportif."""

from __future__ import annotations

import logging
from typing import Literal

from kata_pipeline.competition_formats import normalize_competition_type
from kata_pipeline.detection.kata_duration_ref import KataDurationReference
from kata_pipeline.detection.pairing import MatchPair, PairingResult
from kata_pipeline.schemas.clip import KataClip
from kata_pipeline.schemas.live import LiveRecord
from kata_pipeline.schemas.passage import ExpectedPassage

logger = logging.getLogger(__name__)


def build_expected_passages(
    matches: list[dict],
    live: LiveRecord,
) -> list[ExpectedPassage]:
    """Construire les passages attendus à partir du dataset sportif.

    Chaque match génère 2 passages : rouge (passage_order=1) puis bleu (passage_order=2).

    Args:
        matches: Liste de dictionnaires représentant les matchs
        live: Métadonnées du live associé

    Returns:
        Liste de passages attendus, triés par ordre de passage global
    """
    passages: list[ExpectedPassage] = []

    for match in matches:
        competition_type = normalize_competition_type(
            match.get("competition_type") or live.competition_type,
            match.get("competition"),
        )
        # Déterminer l'issue pour chaque athlète
        winner = match.get("winner", "")
        red_issue = _determine_issue(match["athlete_red"], winner)
        blue_issue = _determine_issue(match["athlete_blue"], winner)

        # Générer un id_match unique
        id_match = _generate_match_id(match, live.id_live)

        # Passage rouge
        passages.append(ExpectedPassage(
            id_match=id_match,
            id_live=live.id_live,
            competition=match["competition"],
            competition_type=competition_type,
            category=match["category"],
            round=match["round"],
            match_order=int(match["match_order"]),
            passage_order=1,
            color="red",
            athlete=match["athlete_red"],
            kata=match["kata_red"],
            style=match.get("style_red"),
            score=_safe_float(match.get("score_red")),
            flag_result=match.get("flag_result_red"),
            issue=red_issue,
            opponent=match["athlete_blue"],
            opponent_kata=match["kata_blue"],
        ))

        # Passage bleu
        passages.append(ExpectedPassage(
            id_match=id_match,
            id_live=live.id_live,
            competition=match["competition"],
            competition_type=competition_type,
            category=match["category"],
            round=match["round"],
            match_order=int(match["match_order"]),
            passage_order=2,
            color="blue",
            athlete=match["athlete_blue"],
            kata=match["kata_blue"],
            style=match.get("style_blue"),
            score=_safe_float(match.get("score_blue")),
            flag_result=match.get("flag_result_blue"),
            issue=blue_issue,
            opponent=match["athlete_red"],
            opponent_kata=match["kata_red"],
        ))

    passages.sort(key=lambda p: p.global_passage_order)
    logger.info("Passages attendus générés: %d (depuis %d matchs)", len(passages), len(matches))
    return passages


def align_passages_with_segments(
    passages: list[ExpectedPassage],
    pairing_result: PairingResult,
    live: LiveRecord,
    kata_ref: KataDurationReference | None = None,
) -> list[KataClip]:
    """Aligner les passages attendus avec les segments détectés.

    Associe chaque passage à son segment vidéo correspondant.
    Si kata_ref est fourni, auto-corrige les bornes quand la durée détectée
    est trop courte par rapport au minimum attendu du kata.

    Args:
        passages: Passages attendus triés
        pairing_result: Résultat du pairing des segments
        live: Métadonnées du live
        kata_ref: Référence de durées de kata (pour auto-correction)

    Returns:
        Liste de KataClip alignés
    """
    clips: list[KataClip] = []

    # Les paires sont en ordre chronologique (match_index = 1, 2, 3...)
    # Les passages sont triés par global_passage_order
    # On mappe la Nème paire au Nème match unique dans les passages
    unique_match_orders = sorted(set(p.match_order for p in passages))
    pair_by_match_order: dict[int, MatchPair] = {}
    for i, pair in enumerate(pairing_result.pairs):
        if i < len(unique_match_orders):
            pair_by_match_order[unique_match_orders[i]] = pair

    for passage in passages:
        pair = pair_by_match_order.get(passage.match_order)

        if pair is None:
            # Pas de segment détecté pour ce match
            clip = _create_clip_no_segment(passage, live)
            clips.append(clip)
            continue

        # Sélectionner le segment rouge ou bleu
        if passage.color == "red":
            segment = pair.red_segment
            # Le segment suivant (bleu) sert de borne max pour l'extension
            next_segment = pair.blue_segment
            prev_segment = None
        else:
            segment = pair.blue_segment
            next_segment = None
            prev_segment = pair.red_segment

        # Auto-correction des bornes si durée trop courte pour le kata
        start_time = segment.start_time
        end_time = segment.end_time
        duration = end_time - start_time

        if kata_ref is not None and passage.kata:
            bounds = kata_ref.get_bounds(passage.kata)
            if bounds is not None and duration < bounds.safe_min:
                deficit = bounds.mean - duration
                # Déterminer dans quel sens étendre
                # Si rouge : on peut étendre la fin (mais pas au-delà du début du bleu)
                # Si bleu : on peut étendre le début (mais pas avant la fin du rouge)
                if passage.color == "red" and next_segment is not None:
                    # Étendre vers la fin (kata coupé avant la fin)
                    max_end = next_segment.start_time - 10.0  # garder 10s de marge
                    new_end = min(end_time + deficit, max_end)
                    if new_end > end_time:
                        logger.info(
                            "Auto-correction %s (rouge): fin étendue %.1f → %.1f (+%.1fs) "
                            "pour atteindre durée min %s",
                            passage.athlete, end_time, new_end,
                            new_end - end_time, passage.kata,
                        )
                        end_time = new_end

                elif passage.color == "blue" and prev_segment is not None:
                    # Étendre vers le début (kata coupé au début)
                    min_start = prev_segment.end_time + 10.0  # garder 10s de marge
                    new_start = max(start_time - deficit, min_start)
                    if new_start < start_time:
                        logger.info(
                            "Auto-correction %s (bleu): début avancé %.1f → %.1f (−%.1fs) "
                            "pour atteindre durée min %s",
                            passage.athlete, start_time, new_start,
                            start_time - new_start, passage.kata,
                        )
                        start_time = new_start

                # Si on n'a pas de segment adjacent, étendre des deux côtés
                elif passage.color == "red":
                    end_time = end_time + deficit * 0.7
                    start_time = max(0.0, start_time - deficit * 0.3)
                    logger.info(
                        "Auto-correction %s: étendu des deux côtés (+%.1fs)",
                        passage.athlete, deficit,
                    )
                else:
                    start_time = max(0.0, start_time - deficit * 0.7)
                    end_time = end_time + deficit * 0.3
                    logger.info(
                        "Auto-correction %s: étendu des deux côtés (+%.1fs)",
                        passage.athlete, deficit,
                    )

        # Calculer la confiance
        confidence = segment.confidence
        needs_review = segment.needs_review or pair.needs_review

        clip = KataClip(
            id_clip=_generate_clip_id(passage),
            id_match=passage.id_match,
            id_live=passage.id_live,
            video_url=live.url,
            source_video_path=live.local_path,
            athlete=passage.athlete,
            color=passage.color,
            kata=passage.kata,
            style=passage.style,
            score=passage.score,
            flag_result=passage.flag_result,
            issue=passage.issue,
            opponent=passage.opponent,
            opponent_kata=passage.opponent_kata,
            competition=passage.competition,
            competition_type=passage.competition_type,
            category=passage.category,
            round=passage.round,
            match_order=passage.match_order,
            passage_order=passage.passage_order,
            start_time=start_time,
            end_time=end_time,
            confidence_score=confidence,
            needs_review=needs_review,
            validation_status="pending" if needs_review else "auto_validated",
        )
        clips.append(clip)

    auto_validated = sum(1 for c in clips if c.validation_status == "auto_validated")
    needs_review = sum(1 for c in clips if c.needs_review)
    logger.info(
        "Alignement terminé: %d clips (%d auto-validés, %d needs_review)",
        len(clips), auto_validated, needs_review,
    )

    return clips


def _create_clip_no_segment(passage: ExpectedPassage, live: LiveRecord) -> KataClip:
    """Créer un clip sans segment détecté (needs_review obligatoire)."""
    return KataClip(
        id_clip=_generate_clip_id(passage),
        id_match=passage.id_match,
        id_live=passage.id_live,
        video_url=live.url,
        source_video_path=live.local_path,
        athlete=passage.athlete,
        color=passage.color,
        kata=passage.kata,
        style=passage.style,
        score=passage.score,
        flag_result=passage.flag_result,
        issue=passage.issue,
        opponent=passage.opponent,
        opponent_kata=passage.opponent_kata,
        competition=passage.competition,
        competition_type=passage.competition_type,
        category=passage.category,
        round=passage.round,
        match_order=passage.match_order,
        passage_order=passage.passage_order,
        start_time=0.0,
        end_time=0.0,
        confidence_score=0.0,
        needs_review=True,
        validation_status="needs_correction",
    )


def _generate_match_id(match: dict, live_id: str) -> str:
    """Générer un identifiant unique pour un match."""
    competition = match["competition"].replace(" ", "_")[:20]
    category = match["category"].replace(" ", "_")[:15]
    round_name = match["round"].replace(" ", "_")[:10]
    order = match["match_order"]
    return f"{live_id}_{competition}_{category}_{round_name}_m{order}"


def _generate_clip_id(passage: ExpectedPassage) -> str:
    """Générer un identifiant unique pour un clip."""
    return (
        f"{passage.id_match}_{passage.color}_"
        f"{passage.athlete.replace(' ', '_')[:20]}"
    )


def _determine_issue(
    athlete: str, winner: str | None
) -> Literal["win", "loss", "unknown"]:
    """Déterminer l'issue pour un athlète."""
    if not winner:
        return "unknown"
    if athlete.strip().lower() == winner.strip().lower():
        return "win"
    return "loss"


def _safe_float(value: object) -> float | None:
    """Convertir en float ou retourner None."""
    if value is None:
        return None
    try:
        f = float(value)  # type: ignore[arg-type]
        import math
        if math.isnan(f):
            return None
        return f
    except (ValueError, TypeError):
        return None
