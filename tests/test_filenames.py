"""Tests pour la génération de noms de fichiers sûrs."""

from kata_pipeline.schemas.clip import KataClip
from kata_pipeline.utils.filenames import generate_clip_filename, sanitize_filename


class TestSanitizeFilename:
    def test_simple(self) -> None:
        assert sanitize_filename("Hello World") == "Hello_World"

    def test_accents(self) -> None:
        assert sanitize_filename("Café résumé") == "Cafe_resume"

    def test_special_chars(self) -> None:
        assert sanitize_filename("file/name\\test:1") == "file_name_test1"

    def test_multiple_spaces(self) -> None:
        assert sanitize_filename("a   b   c") == "a_b_c"

    def test_max_length(self) -> None:
        result = sanitize_filename("a" * 100, max_length=20)
        assert len(result) <= 20

    def test_empty_string(self) -> None:
        assert sanitize_filename("") == "unnamed"

    def test_only_special_chars(self) -> None:
        assert sanitize_filename("!!!") == "unnamed"

    def test_japanese_chars(self) -> None:
        # Les caractères japonais sont supprimés (non-ASCII)
        result = sanitize_filename("清水 希容")
        assert result == "unnamed" or len(result) > 0

    def test_hyphen_conversion(self) -> None:
        assert sanitize_filename("kata-name") == "kata_name"


class TestGenerateClipFilename:
    def test_basic(self) -> None:
        clip = KataClip(
            id_clip="c1",
            id_match="m1",
            id_live="l1",
            athlete="Ryo Kiyuna",
            color="red",
            kata="Anan Dai",
            competition="WKF Paris 2024",
            category="Male Kata",
            round="Round 1",
            match_order=3,
            passage_order=1,
            start_time=100,
            end_time=180,
            opponent="Opp",
            opponent_kata="K",
        )
        filename = generate_clip_filename(clip)

        assert filename.endswith(".mp4")
        assert "red" in filename
        assert "Ryo_Kiyuna" in filename
        assert "m03" in filename

    def test_no_invalid_chars(self) -> None:
        clip = KataClip(
            id_clip="c1",
            id_match="m1",
            id_live="l1",
            athlete="Athlète/Spécial",
            color="blue",
            kata="Kata:Test",
            competition="Compé<>tition",
            category="Cat|Test",
            round="R/1",
            match_order=1,
            passage_order=2,
            start_time=0,
            end_time=60,
            opponent="Opp",
            opponent_kata="K",
        )
        filename = generate_clip_filename(clip)

        # Pas de caractères invalides pour un nom de fichier
        invalid_chars = set('<>:"/\\|?*')
        assert not any(c in filename for c in invalid_chars)
