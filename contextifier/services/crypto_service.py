# contextifier/services/crypto_service.py
"""
Password-protected MS Office file decryption service.

Uses ``msoffcrypto-tool`` to detect and decrypt password-protected
DOCX, PPTX, XLSX, DOC, PPT, XLS files.

Supported formats:
- OOXML: .docx, .pptx, .xlsx (AES-encrypted ZIP)
- OLE2: .doc, .ppt, .xls (RC4 / AES)

Usage::

    from contextifier.services.crypto_service import decrypt_if_encrypted

    # Auto-detect encryption and decrypt with password
    data = decrypt_if_encrypted(file_bytes, password="secret")

    # Returns original data if not encrypted
    data = decrypt_if_encrypted(file_bytes)
"""

from __future__ import annotations

import io
import logging
from typing import Optional

logger = logging.getLogger("contextifier.crypto")


def is_encrypted(file_data: bytes) -> bool:
    """Check if a file is password-protected.

    Args:
        file_data: Raw file bytes.

    Returns:
        True if the file is encrypted.
    """
    try:
        import msoffcrypto
    except ImportError:
        return False

    stream = io.BytesIO(file_data)
    try:
        f = msoffcrypto.OfficeFile(stream)
        return f.is_encrypted()
    except Exception:
        return False


def decrypt_if_encrypted(
    file_data: bytes,
    *,
    password: Optional[str] = None,
) -> bytes:
    """Decrypt file data if it is password-protected.

    If the file is not encrypted, returns the original data unchanged.
    If encrypted and no password (or wrong password) is provided,
    raises ``EncryptedFileError``.

    Args:
        file_data: Raw file bytes.
        password: Password for decryption (None = try empty password).

    Returns:
        Decrypted file bytes, or original if not encrypted.

    Raises:
        EncryptedFileError: If file is encrypted and cannot be decrypted.
    """
    try:
        import msoffcrypto
    except ImportError:
        return file_data

    stream = io.BytesIO(file_data)
    try:
        f = msoffcrypto.OfficeFile(stream)
        if not f.is_encrypted():
            return file_data
    except Exception:
        return file_data

    # File is encrypted — attempt decryption
    from contextifier.errors import FileReadError

    pw = password if password is not None else ""
    try:
        f.load_key(password=pw)
        output = io.BytesIO()
        f.decrypt(output)
        logger.info("Successfully decrypted password-protected file")
        return output.getvalue()
    except Exception as e:
        raise FileReadError(
            "File is password-protected and could not be decrypted",
            context={"has_password": password is not None},
            cause=e,
        )
