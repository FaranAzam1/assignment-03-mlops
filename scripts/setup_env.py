"""Create .env and airflow-passwords.json with random local credentials (both git-ignored)."""

from __future__ import annotations

import json
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    env = ROOT / ".env"
    if env.exists():
        print(".env already exists; leaving it unchanged")
        return
    password, jwt = secrets.token_urlsafe(12), secrets.token_urlsafe(32)
    lines = (ROOT / ".env.example").read_text().splitlines()
    out = []
    for line in lines:
        if line.startswith("AIRFLOW_PASSWORD="):
            line = f"AIRFLOW_PASSWORD={password}"
        elif line.startswith("POSTGRES_PASSWORD="):
            line = f"POSTGRES_PASSWORD={secrets.token_urlsafe(12)}"
        elif line.startswith("AIRFLOW_JWT_SECRET="):
            line = f"AIRFLOW_JWT_SECRET={jwt}"
        out.append(line)
    env.write_text("\n".join(out) + "\n")
    (ROOT / "airflow-passwords.json").write_text(json.dumps({"admin": password}))
    print("created .env and airflow-passwords.json (random credentials, not committed)")


if __name__ == "__main__":
    main()
