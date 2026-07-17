from __future__ import annotations

import hmac

from cryptography.fernet import Fernet, InvalidToken


class VectorCipher:
    def __init__(self, key: str) -> None:
        try:
            self._fernet = Fernet(key.encode("ascii"))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("TWINLENS_FERNET_KEY must be a valid Fernet key") from exc

    def encrypt(self, value: bytes) -> bytes:
        return self._fernet.encrypt(value)

    def decrypt(self, value: bytes) -> bytes:
        try:
            return self._fernet.decrypt(value)
        except InvalidToken as exc:
            raise RuntimeError("Could not decrypt stored embedding; verify the Fernet key") from exc


def valid_api_token(expected: str, supplied: str | None) -> bool:
    if supplied is None:
        return False
    return hmac.compare_digest(expected.encode("utf-8"), supplied.encode("utf-8"))
