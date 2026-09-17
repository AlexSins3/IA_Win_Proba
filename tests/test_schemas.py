"""Tests pour la validation des schémas de données."""

import pytest
from pydantic import ValidationError

from kata_pipeline.schemas.clip import KataClip
from kata_pipeline.schemas.competition import MatchRecord
from kata_pipeline.schemas.live import LiveRecord
from kata_pipeline.schemas.passage import ExpectedPassage


class TestMatchRecord:
    def test_valid_minimal(self) -> None:
        record = MatchRecord(
            competition="WKF Paris 2024",
            category="Male Kata",
            round="Round 1",
            match_order=1,
            athlete_red="Kiyuna",
            athlete_blue="Quintero",
            kata_red="Anan",
            kata_blue="Kushanku",
        )
        assert record.competition == "WKF Paris 2024"
        assert record.match_order == 1

    def test_match_order_coercion(self) -> None:
        record = MatchRecord(
            competition="Test",
            category="Test",
            round="R1",
            match_order="3",  # type: ignore
            athlete_red="A",
            athlete_blue="B",
            kata_red="K1",
            kata_blue="K2",
        )
        assert record.match_order == 3

    def test_invalid_missing_required(self) -> None:
        with pytest.raises(ValidationError):
            MatchRecord(
                competition="Test",
                category="Test",
                # round manquant
                match_order=1,
                athlete_red="A",
                athlete_blue="B",
                kata_red="K1",
                kata_blue="K2",
            )  # type: ignore


class TestLiveRecord:
    def test_valid_minimal(self) -> None:
        live = LiveRecord(
            id_live="live_001",
            competition="WKF Paris",
            category="Male Kata",
        )
        assert live.status == "pending"
        assert live.analysis_quality == "low"

    def test_with_all_fields(self) -> None:
        live = LiveRecord(
            id_live="live_002",
            url="https://youtube.com/watch?v=test",
            competition="WKF Tokyo",
            category="Female Kata",
            useful_start=120.0,
            useful_end=7200.0,
            status="downloaded",
        )
        assert live.useful_start == 120.0


class TestExpectedPassage:
    def test_global_passage_order(self) -> None:
        passage = ExpectedPassage(
            id_match="m1",
            id_live="l1",
            competition="Test",
            category="Cat",
            round="R1",
            match_order=2,
            passage_order=1,  # rouge du match 2
            color="red",
            athlete="Athlete",
            kata="Anan",
            opponent="Opponent",
            opponent_kata="Kushanku",
        )
        # (2-1) * 2 + 1 = 3
        assert passage.global_passage_order == 3

    def test_blue_passage_order(self) -> None:
        passage = ExpectedPassage(
            id_match="m1",
            id_live="l1",
            competition="Test",
            category="Cat",
            round="R1",
            match_order=1,
            passage_order=2,  # bleu du match 1
            color="blue",
            athlete="Athlete",
            kata="Anan",
            opponent="Opponent",
            opponent_kata="Kushanku",
        )
        # (1-1) * 2 + 2 = 2
        assert passage.global_passage_order == 2


class TestKataClip:
    def test_duration_computed(self) -> None:
        clip = KataClip(
            id_clip="c1",
            id_match="m1",
            id_live="l1",
            athlete="Test",
            color="red",
            kata="Anan",
            competition="Comp",
            category="Cat",
            round="R1",
            match_order=1,
            passage_order=1,
            start_time=100.0,
            end_time=180.0,
            opponent="Opp",
            opponent_kata="KataOpp",
        )
        assert clip.duration == 80.0

    def test_default_status(self) -> None:
        clip = KataClip(
            id_clip="c1",
            id_match="m1",
            id_live="l1",
            athlete="Test",
            color="blue",
            kata="Anan",
            competition="Comp",
            category="Cat",
            round="R1",
            match_order=1,
            passage_order=2,
            start_time=0,
            end_time=60,
            opponent="Opp",
            opponent_kata="KataOpp",
        )
        assert clip.validation_status == "pending"
        assert clip.needs_review is False
