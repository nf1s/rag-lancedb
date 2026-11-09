"""Password hashing using the standard library's scrypt implementation."""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_DKLEN = 64


def hash_password(password: str) -> str:
    """Return a self-describing scrypt hash: ``scrypt$N$r$p$salt$digest``."""
    if not password:
        raise ValueError("password must not be empty")
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_DKLEN,
    )
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    """Constant-time verification of a hash produced by :func:`hash_password`."""
    try:
        scheme, n, r, p, salt_b64, digest_b64 = encoded.split("$")
    except (AttributeError, ValueError):
        return False
    if scheme != "scrypt":
        return False
    try:
        salt = _unb64(salt_b64)
        expected = _unb64(digest_b64)
        computed = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(expected),
        )
    except (TypeError, ValueError):
        return False
    return hmac.compare_digest(computed, expected)


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
