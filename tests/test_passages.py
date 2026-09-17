"""Tests pour la génération des passages attendus."""

import pytest

from kata_pipeline.alignment.aligner import build_expected_passages
from kata_pipeline.schemas.live import LiveRecord


@pytest.fixture
def sample_live() -> LiveRecord:
    return LiveRecord(
        id_live="live_001",
        competition="WKF Premier League Paris 2024",
        category="Male Kata",
    )


@pytest.fixture
def sample_matches() -> list[dict]:
    return [
        {
            "competition": "WKF Premier League Paris 2024",
            "category": "Male Kata",
            "round": "Round 1",
            "match_order": 1,
            "athlete_red": "Ryo Kiyuna",
            "athlete_blue": "Damian Quintero",
            "kata_red": "Anan",
            "kata_blue": "Chatanyara Kushanku",
            "score_red": 27.5,
            "score_blue": 26.8,
            "winner": "Ryo Kiyuna",
        },
        {
            "competition": "WKF Premier League Paris 2024",
            "category": "Male Kata",
            "round": "Round 1",
            "match_order": 2,
            "athlete_red": "Antonio Diaz",
            "athlete_blue": "Kazumasa Moto",
            "kata_red": "Suparinpei",
            "kata_blue": "Ohan Dai",
            "winner": "Kazumasa Moto",
        },
    ]


class TestBuildExpectedPassages:
    def test_generates_two_passages_per_match(
        self, sample_matches: list[dict], sample_live: LiveRecord
    ) -> None:
        passages = build_expected_passages(sample_matches, sample_live)
        assert len(passages) == 4  # 2 matchs × 2 passages

    def test_passage_order(
        self, sample_matches: list[dict], sample_live: LiveRecord
    ) -> None:
        passages = build_expected_passages(sample_matches, sample_live)
        # Les passages doivent être triés par ordre global
        orders = [p.global_passage_order for p in passages]
        assert orders == sorted(orders)

    def test_red_before_blue(
        self, sample_matches: list[dict], sample_live: LiveRecord
    ) -> None:
        passages = build_expected_passages(sample_matches, sample_live)
        # Pour chaque match, rouge (passage_order=1) avant bleu (passage_order=2)
        match1_passages = [p for p in passages if p.match_order == 1]
        assert match1_passages[0].color == "red"
        assert match1_passages[1].color == "blue"

    def test_issue_determined_correctly(
        self, sample_matches: list[dict], sample_live: LiveRecord
    ) -> None:
        passages = build_expected_passages(sample_matches, sample_live)
        # Match 1: Kiyuna gagne
        red_m1 = next(p for p in passages if p.match_order == 1 and p.color == "red")
        blue_m1 = next(p for p in passages if p.match_order == 1 and p.color == "blue")
        assert red_m1.issue == "win"
        assert blue_m1.issue == "loss"

    def test_opponent_is_set(
        self, sample_matches: list[dict], sample_live: LiveRecord
    ) -> None:
        passages = build_expected_passages(sample_matches, sample_live)
        red_m1 = next(p for p in passages if p.match_order == 1 and p.color == "red")
        assert red_m1.opponent == "Damian Quintero"
        assert red_m1.opponent_kata == "Chatanyara Kushanku"

    def test_live_id_propagated(
        self, sample_matches: list[dict], sample_live: LiveRecord
    ) -> None:
        passages = build_expected_passages(sample_matches, sample_live)
        for p in passages:
            assert p.id_live == "live_001"

    def test_empty_matches(self, sample_live: LiveRecord) -> None:
        passages = build_expected_passages([], sample_live)
        assert passages == []
