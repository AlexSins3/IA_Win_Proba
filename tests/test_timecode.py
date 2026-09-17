"""Tests pour les conversions de timecode."""

import pytest

from kata_pipeline.utils.timecode import seconds_to_timecode, timecode_to_seconds


class TestTimecodeToSeconds:
    def test_hhmmss_zero(self) -> None:
        assert timecode_to_seconds("00:00:00") == 0.0

    def test_hhmmss_simple(self) -> None:
        assert timecode_to_seconds("01:30:00") == 5400.0

    def test_hhmmss_full(self) -> None:
        assert timecode_to_seconds("02:15:45") == 8145.0

    def test_mmss_format(self) -> None:
        assert timecode_to_seconds("05:30") == 330.0

    def test_with_milliseconds(self) -> None:
        result = timecode_to_seconds("01:02:03.500")
        assert result == pytest.approx(3723.5)

    def test_with_spaces(self) -> None:
        assert timecode_to_seconds("  01:00:00  ") == 3600.0

    def test_invalid_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Format de timecode invalide"):
            timecode_to_seconds("invalid")

    def test_single_number_raises(self) -> None:
        with pytest.raises(ValueError):
            timecode_to_seconds("123")


class TestSecondsToTimecode:
    def test_zero(self) -> None:
        assert seconds_to_timecode(0.0) == "00:00:00"

    def test_one_hour(self) -> None:
        assert seconds_to_timecode(3600.0) == "01:00:00"

    def test_complex(self) -> None:
        assert seconds_to_timecode(8145.0) == "02:15:45"

    def test_with_milliseconds(self) -> None:
        result = seconds_to_timecode(3723.5, include_ms=True)
        assert result == "01:02:03.500"

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="négatif"):
            seconds_to_timecode(-1.0)

    def test_roundtrip(self) -> None:
        original = "01:23:45"
        seconds = timecode_to_seconds(original)
        result = seconds_to_timecode(seconds)
        assert result == original
