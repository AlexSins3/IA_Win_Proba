"""Tests pour la logique de pairing des segments."""

import pytest

from kata_pipeline.config import PairingConfig
from kata_pipeline.detection.pairing import pair_segments
from kata_pipeline.detection.segment_detector import CandidateSegment


def _make_segment(start: float, end: float, confidence: float = 0.8) -> CandidateSegment:
    """Helper pour créer un segment candidat."""
    return CandidateSegment(
        start_time=start,
        end_time=end,
        mean_motion=20.0,
        max_motion=40.0,
        confidence=confidence,
    )


class TestPairSegments:
    def test_perfect_pairing(self) -> None:
        """4 segments pour 2 matchs = pairing parfait."""
        segments = [
            _make_segment(10, 80),   # rouge match 1
            _make_segment(100, 170),  # bleu match 1
            _make_segment(250, 320),  # rouge match 2
            _make_segment(350, 420),  # bleu match 2
        ]
        result = pair_segments(segments, expected_match_count=2)

        assert result.detected_matches == 2
        assert not result.needs_review
        assert len(result.pairs) == 2
        assert result.pairs[0].match_index == 1
        assert result.pairs[1].match_index == 2

    def test_segment_count_mismatch(self) -> None:
        """Moins de segments que prévu -> needs_review."""
        segments = [
            _make_segment(10, 80),
            _make_segment(100, 170),
        ]
        result = pair_segments(segments, expected_match_count=2)

        assert result.needs_review
        assert result.detected_matches == 1

    def test_overlap_detection(self) -> None:
        """Chevauchement rouge-bleu détecté."""
        segments = [
            _make_segment(10, 100),
            _make_segment(90, 170),  # Commence avant la fin du précédent
        ]
        result = pair_segments(segments, expected_match_count=1)

        assert result.pairs[0].needs_review
        assert any("Chevauchement" in r for r in result.pairs[0].review_reasons or [])

    def test_large_gap_within_match(self) -> None:
        """Écart trop grand entre rouge et bleu."""
        config = PairingConfig(max_gap_within_match=60.0)
        segments = [
            _make_segment(10, 80),
            _make_segment(200, 270),  # Gap de 120s > 60s max
        ]
        result = pair_segments(segments, expected_match_count=1, config=config)

        assert result.pairs[0].needs_review
        assert any("trop grand" in r for r in result.pairs[0].review_reasons or [])

    def test_empty_segments(self) -> None:
        """Aucun segment -> 0 paires."""
        result = pair_segments([], expected_match_count=3)

        assert result.detected_matches == 0
        assert result.needs_review

    def test_extra_segments(self) -> None:
        """Plus de segments que prévu."""
        segments = [
            _make_segment(10, 80),
            _make_segment(100, 170),
            _make_segment(250, 320),
            _make_segment(350, 420),
            _make_segment(500, 570),  # Extra
            _make_segment(600, 670),  # Extra
        ]
        result = pair_segments(segments, expected_match_count=2)

        # On crée 2 paires avec les 4 premiers, les 2 derniers sont unmatched
        assert result.detected_matches == 2
        assert len(result.unmatched_segments) == 2

    def test_duration_ratio_alert(self) -> None:
        """Ratio de durée anormal entre rouge et bleu."""
        config = PairingConfig(duration_ratio_tolerance=0.3)
        segments = [
            _make_segment(10, 80),    # 70s
            _make_segment(100, 120),  # 20s - ratio = 20/70 = 0.28 < 0.3
        ]
        result = pair_segments(segments, expected_match_count=1, config=config)

        assert result.pairs[0].needs_review
        assert any("Ratio" in r for r in result.pairs[0].review_reasons or [])
