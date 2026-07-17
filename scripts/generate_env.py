from __future__ import annotations

import base64
import os
import secrets
from pathlib import Path


path = Path(".env")
if path.exists():
    raise SystemExit(".env already exists; refusing to overwrite it")

fernet_key = base64.urlsafe_b64encode(os.urandom(32)).decode("ascii")
api_token = secrets.token_urlsafe(32)
path.write_text(
    f"TWINLENS_API_TOKEN={api_token}\nTWINLENS_FERNET_KEY={fernet_key}\n",
    encoding="utf-8",
)
print("Created .env with a random API token and encryption key. Keep it private and back it up.")
