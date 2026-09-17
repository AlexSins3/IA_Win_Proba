"""Tests des formats SA/K1 et du routage des dossiers vidéo."""

from pathlib import Path

import pytest

from kata_pipeline.competition_formats import (
    competition_type_from_path,
    get_competition_format,
    normalize_competition_type,
    typed_competition_dir,
    typed_stage_dir,
)


def test_k1_rounds_are_centralized() -> None:
    profile = get_competition_format("K1")
    assert profile.pool_rounds == ("Pool_1", "Pool_2", "Pool_3")
    assert profile.finals_rounds == ("R1", "R2", "Bronze", "Final")
    assert profile.finals_labels["R2"] == "r2"


def test_type_is_inferred_from_legacy_competition_name() -> None:
    assert normalize_competition_type(None, "SA_ACoruna") == "SA"
    assert normalize_competition_type(None, "K1_Paris") == "K1"
    with pytest.raises(ValueError, match="inconnu"):
        normalize_competition_type("WKF")


def test_typed_clip_directories_keep_legacy_compatibility() -> None:
    assert typed_competition_dir(Path("data/clips"), "SA") == Path("data/clips/SA")
    assert typed_competition_dir(Path("data/clips/SA"), "K1") == Path("data/clips/K1")
    assert typed_stage_dir(Path("data/clips/pending"), "K1") == Path(
        "data/clips/K1/pending"
    )
    assert typed_stage_dir(Path("data/clips/SA/pending"), "K1") == Path(
        "data/clips/K1/pending"
    )
    assert competition_type_from_path(
        Path("data/clips/SA/pool_1/a.mp4"), Path("data/clips")
    ) == "SA"
    assert competition_type_from_path(
        Path("data/clips/pool_1/a.mp4"), Path("data/clips")
    ) is None
