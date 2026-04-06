# tests/unit/test_encoding_config.py
"""Tests for EncodingConfig (P6-7)."""

from __future__ import annotations

import pytest

from contextifier.config import EncodingConfig, ProcessingConfig


class TestEncodingConfigDefaults:
    """Tests for EncodingConfig default values."""

    def test_default_fallback_encodings(self):
        cfg = EncodingConfig()
        assert cfg.fallback_encodings == (
            "utf-8", "utf-8-sig", "cp949", "euc-kr", "latin-1", "ascii",
        )

    def test_default_force_encoding_none(self):
        cfg = EncodingConfig()
        assert cfg.force_encoding is None

    def test_default_min_confidence(self):
        cfg = EncodingConfig()
        assert cfg.min_confidence == 0.7

    def test_is_frozen(self):
        cfg = EncodingConfig()
        with pytest.raises(AttributeError):
            cfg.force_encoding = "utf-8"  # type: ignore[misc]


class TestEncodingConfigCustom:
    """Tests for custom EncodingConfig values."""

    def test_custom_fallback_encodings(self):
        cfg = EncodingConfig(fallback_encodings=("shift_jis", "utf-8"))
        assert cfg.fallback_encodings == ("shift_jis", "utf-8")

    def test_force_encoding(self):
        cfg = EncodingConfig(force_encoding="cp949")
        assert cfg.force_encoding == "cp949"

    def test_custom_min_confidence(self):
        cfg = EncodingConfig(min_confidence=0.9)
        assert cfg.min_confidence == 0.9


class TestProcessingConfigEncoding:
    """Tests for EncodingConfig integration in ProcessingConfig."""

    def test_default_encoding_in_processing_config(self):
        config = ProcessingConfig()
        assert isinstance(config.encoding, EncodingConfig)
        assert config.encoding.force_encoding is None

    def test_with_encoding_fluent_method(self):
        config = ProcessingConfig()
        config2 = config.with_encoding(force_encoding="utf-8")
        assert config2.encoding.force_encoding == "utf-8"
        assert config.encoding.force_encoding is None  # original unchanged

    def test_with_encoding_fallback(self):
        config = ProcessingConfig().with_encoding(
            fallback_encodings=("shift_jis", "utf-8")
        )
        assert config.encoding.fallback_encodings == ("shift_jis", "utf-8")

    def test_serialization_roundtrip(self):
        config = ProcessingConfig().with_encoding(
            force_encoding="cp949",
            min_confidence=0.5,
            fallback_encodings=("utf-8", "latin-1"),
        )
        d = config.to_dict()
        assert d["encoding"]["force_encoding"] == "cp949"
        assert d["encoding"]["min_confidence"] == 0.5

        restored = ProcessingConfig.from_dict(d)
        assert restored.encoding.force_encoding == "cp949"
        assert restored.encoding.min_confidence == 0.5
        assert restored.encoding.fallback_encodings == ("utf-8", "latin-1")


class TestConverterEncodingConfig:
    """Tests for EncodingConfig integration in converters."""

    def test_csv_converter_uses_force_encoding(self):
        from contextifier.handlers.csv.converter import CsvConverter
        enc_cfg = EncodingConfig(force_encoding="cp949")
        converter = CsvConverter(encoding_config=enc_cfg)
        assert converter._encodings == ["cp949"]

    def test_csv_converter_uses_fallback_encodings(self):
        from contextifier.handlers.csv.converter import CsvConverter
        enc_cfg = EncodingConfig(fallback_encodings=("shift_jis", "utf-8"))
        converter = CsvConverter(encoding_config=enc_cfg)
        assert converter._encodings == ["shift_jis", "utf-8"]

    def test_csv_converter_explicit_encodings_override_config(self):
        from contextifier.handlers.csv.converter import CsvConverter
        enc_cfg = EncodingConfig(fallback_encodings=("shift_jis",))
        converter = CsvConverter(encodings=["euc-kr"], encoding_config=enc_cfg)
        # Explicit encodings take priority over config
        assert converter._encodings == ["euc-kr"]

    def test_text_converter_uses_force_encoding(self):
        from contextifier.handlers.text.converter import TextConverter
        enc_cfg = EncodingConfig(force_encoding="utf-16")
        converter = TextConverter(encoding_config=enc_cfg)
        assert converter._encodings == ["utf-16"]

    def test_text_converter_uses_fallback_encodings(self):
        from contextifier.handlers.text.converter import TextConverter
        enc_cfg = EncodingConfig(fallback_encodings=("ascii", "utf-8"))
        converter = TextConverter(encoding_config=enc_cfg)
        assert converter._encodings == ["ascii", "utf-8"]

    def test_text_converter_no_config_uses_default(self):
        from contextifier.handlers.text.converter import TextConverter, DEFAULT_ENCODINGS
        converter = TextConverter()
        assert converter._encodings == DEFAULT_ENCODINGS
