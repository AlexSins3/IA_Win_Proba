"""Tests du générateur de CSV pour les formats SA et K1."""

import csv

from scripts.build_pool_dataset import (
    LIVES_HEADER,
    append_finals_lives,
    append_lives,
    build_dataset,
    pair_rows,
    split_into_pools,
)


def _row(round_name: str, belt: str, athlete: str) -> dict[str, str]:
    return {
        "N_Tour": round_name,
        "Ceinture": belt,
        "Nom": athlete,
        "Kata": "Anan",
        "Style": "Shito",
        "Drapeau": "5",
        "Victoire": "VRAI" if belt == "R" else "FAUX",
    }


def test_k1_blue_round_typo_does_not_create_a_fake_pool() -> None:
    rows = [
        _row("Pool_1", "R", "A"),
        _row("Pool_1", "B", "B"),
        _row("Pool_2", "R", "A"),
        _row("Pool_1", "B", "C"),  # anomalie réellement présente dans la source
        _row("Pool_2", "R", "B"),
        _row("Pool_2", "B", "C"),
    ]

    pools = split_into_pools(rows, ["Pool_1", "Pool_2", "Pool_3"])
    matches = pair_rows(pools[0], ["Pool_1", "Pool_2", "Pool_3"])

    assert len(pools) == 1
    assert len(matches["Pool_1"]) == 1
    assert len(matches["Pool_2"]) == 2


def test_k1_output_carries_type_and_round_slug() -> None:
    matches = {"Pool_1": [(_row("Pool_1", "R", "A"), _row("Pool_1", "B", "B"))]}
    rows = build_dataset(
        matches,
        "K1_Paris",
        "Female Kata",
        2,
        None,
        None,
        ["Pool_1", "Pool_2", "Pool_3"],
        "K1",
    )

    assert rows[0][14] == "pool_2_pool_1"
    assert rows[0][15] == "K1"


def _write_lives_without_final_newline(path) -> None:
    existing_row = [
        "sa_test_2026_f_pool1_t1",
        "https://youtube.com/watch?v=source1",
        "data/input/source1.mp4",
        "SA_Test",
        "Female Kata",
        "pool_1_t1",
        "10",
        "20",
        "low",
        "high",
        "pending",
        "SA",
    ]
    path.write_text(
        ",".join(LIVES_HEADER) + "\r\n" + ",".join(existing_row),
        encoding="utf-8",
    )


def _read_lives(path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_append_lives_separates_row_when_file_has_no_final_newline(tmp_path) -> None:
    lives_path = tmp_path / "lives.csv"
    _write_lives_without_final_newline(lives_path)

    append_lives(
        lives_path=lives_path,
        competition="SA_Test",
        year="2026",
        sex="F",
        pool_number=2,
        url="https://youtube.com/watch?v=source2",
        video_name="source2.mp4",
        ranges={"T1": (30, 40)},
        rounds_present=["T1"],
        rounds=["T1"],
        competition_type="SA",
    )

    rows = _read_lives(lives_path)
    assert [row["id_live"] for row in rows] == [
        "sa_test_2026_f_pool1_t1",
        "sa_test_2026_f_pool2_t1",
    ]
    assert all(None not in row for row in rows)


def test_append_finals_separates_row_when_file_has_no_final_newline(tmp_path) -> None:
    lives_path = tmp_path / "lives.csv"
    _write_lives_without_final_newline(lives_path)

    append_finals_lives(
        lives_path=lives_path,
        competition="SA_Test",
        year="2026",
        sex="F",
        matches_info=[
            ("quart_1", "https://youtube.com/watch?v=source2", (30, 40)),
        ],
        competition_type="SA",
    )

    rows = _read_lives(lives_path)
    assert [row["id_live"] for row in rows] == [
        "sa_test_2026_f_pool1_t1",
        "sa_test_2026_f_final_quart_1",
    ]
    assert all(None not in row for row in rows)
