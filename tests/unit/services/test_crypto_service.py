# tests/unit/services/test_crypto_service.py
"""Tests for the password-protected file decryption service."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest

from contextifier.services.crypto_service import decrypt_if_encrypted


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

UNENCRYPTED_BYTES = b"plain content data"
DECRYPTED_BYTES = b"decrypted content data"


def _make_mock_office_file(*, encrypted: bool = True, decrypt_ok: bool = True):
    """Create a mock msoffcrypto.OfficeFile instance."""
    mock_file = MagicMock()
    mock_file.is_encrypted.return_value = encrypted

    def _decrypt(output: io.BytesIO):
        if decrypt_ok:
            output.write(DECRYPTED_BYTES)
        else:
            raise Exception("Decryption failed: wrong password")

    mock_file.decrypt.side_effect = _decrypt
    return mock_file


# ═══════════════════════════════════════════════════════════════════════════
# is_encrypted
# ═══════════════════════════════════════════════════════════════════════════


class TestIsEncrypted:
    """Tests for is_encrypted()."""

    @patch("contextifier.services.crypto_service.msoffcrypto", create=True)
    def test_returns_true_for_encrypted_file(self, mock_module):
        mock_office = _make_mock_office_file(encrypted=True)
        mock_module.OfficeFile.return_value = mock_office

        with patch.dict(
            "sys.modules", {"msoffcrypto": mock_module}
        ):
            # Need to reimport to pick up the mock
            import importlib
            import contextifier.services.crypto_service as cs
            importlib.reload(cs)
            assert cs.is_encrypted(UNENCRYPTED_BYTES) is True

    @patch("contextifier.services.crypto_service.msoffcrypto", create=True)
    def test_returns_false_for_unencrypted_file(self, mock_module):
        mock_office = _make_mock_office_file(encrypted=False)
        mock_module.OfficeFile.return_value = mock_office

        with patch.dict(
            "sys.modules", {"msoffcrypto": mock_module}
        ):
            import importlib
            import contextifier.services.crypto_service as cs
            importlib.reload(cs)
            assert cs.is_encrypted(UNENCRYPTED_BYTES) is False

    def test_returns_false_when_msoffcrypto_not_installed(self):
        with patch.dict("sys.modules", {"msoffcrypto": None}):
            import importlib
            import contextifier.services.crypto_service as cs
            importlib.reload(cs)
            assert cs.is_encrypted(UNENCRYPTED_BYTES) is False

    @patch("contextifier.services.crypto_service.msoffcrypto", create=True)
    def test_returns_false_on_parse_error(self, mock_module):
        mock_module.OfficeFile.side_effect = Exception("Not an Office file")

        with patch.dict("sys.modules", {"msoffcrypto": mock_module}):
            import importlib
            import contextifier.services.crypto_service as cs
            importlib.reload(cs)
            assert cs.is_encrypted(UNENCRYPTED_BYTES) is False


# ═══════════════════════════════════════════════════════════════════════════
# decrypt_if_encrypted
# ═══════════════════════════════════════════════════════════════════════════


class TestDecryptIfEncrypted:
    """Tests for decrypt_if_encrypted()."""

    def test_returns_original_if_not_encrypted(self):
        """Unencrypted data passes through unchanged."""
        result = decrypt_if_encrypted(UNENCRYPTED_BYTES)
        assert result == UNENCRYPTED_BYTES

    def test_returns_original_when_msoffcrypto_not_installed(self):
        """Without msoffcrypto, data passes through unchanged."""
        with patch.dict("sys.modules", {"msoffcrypto": None}):
            import importlib
            import contextifier.services.crypto_service as cs
            importlib.reload(cs)
            result = cs.decrypt_if_encrypted(UNENCRYPTED_BYTES, password="secret")
            assert result == UNENCRYPTED_BYTES

    @patch("contextifier.services.crypto_service.msoffcrypto", create=True)
    def test_decrypts_with_password(self, mock_module):
        """Encrypted file is decrypted when correct password is given."""
        mock_office = _make_mock_office_file(encrypted=True, decrypt_ok=True)
        mock_module.OfficeFile.return_value = mock_office

        with patch.dict("sys.modules", {"msoffcrypto": mock_module}):
            import importlib
            import contextifier.services.crypto_service as cs
            importlib.reload(cs)
            result = cs.decrypt_if_encrypted(UNENCRYPTED_BYTES, password="secret")
            assert result == DECRYPTED_BYTES
            mock_office.load_key.assert_called_once_with(password="secret")

    @patch("contextifier.services.crypto_service.msoffcrypto", create=True)
    def test_tries_empty_password_when_none(self, mock_module):
        """When password=None, an empty string is used."""
        mock_office = _make_mock_office_file(encrypted=True, decrypt_ok=True)
        mock_module.OfficeFile.return_value = mock_office

        with patch.dict("sys.modules", {"msoffcrypto": mock_module}):
            import importlib
            import contextifier.services.crypto_service as cs
            importlib.reload(cs)
            result = cs.decrypt_if_encrypted(UNENCRYPTED_BYTES, password=None)
            assert result == DECRYPTED_BYTES
            mock_office.load_key.assert_called_once_with(password="")

    @patch("contextifier.services.crypto_service.msoffcrypto", create=True)
    def test_raises_file_read_error_on_wrong_password(self, mock_module):
        """Wrong password raises FileReadError."""
        mock_office = _make_mock_office_file(encrypted=True, decrypt_ok=False)
        mock_module.OfficeFile.return_value = mock_office

        with patch.dict("sys.modules", {"msoffcrypto": mock_module}):
            import importlib
            import contextifier.services.crypto_service as cs
            importlib.reload(cs)
            from contextifier.errors import FileReadError
            with pytest.raises(FileReadError, match="password-protected"):
                cs.decrypt_if_encrypted(UNENCRYPTED_BYTES, password="wrong")

    @patch("contextifier.services.crypto_service.msoffcrypto", create=True)
    def test_returns_original_if_office_parse_fails(self, mock_module):
        """If msoffcrypto can't parse the file, return original data."""
        mock_module.OfficeFile.side_effect = Exception("Not an Office file")

        with patch.dict("sys.modules", {"msoffcrypto": mock_module}):
            import importlib
            import contextifier.services.crypto_service as cs
            importlib.reload(cs)
            result = cs.decrypt_if_encrypted(UNENCRYPTED_BYTES)
            assert result == UNENCRYPTED_BYTES
