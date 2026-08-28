"""Resolve Apify API token from env, auth.json, or Windows credential store."""

from __future__ import annotations

import json
import os
from pathlib import Path


def resolve_token() -> str | None:
    token = os.environ.get("APIFY_TOKEN") or os.environ.get("APIFY_API_TOKEN")
    if token:
        return token.strip()

    for p in (
        Path.home() / ".apify" / "auth.json",
        Path(os.environ.get("APPDATA", "")) / "apify-cli" / "auth.json",
    ):
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for key in ("token", "apiToken", "accessToken"):
            if data.get(key):
                return str(data[key]).strip()

    # Apify CLI stores the token in the OS keyring (Windows Credential Manager).
    try:
        import win32cred  # type: ignore

        for cred in win32cred.CredEnumerate(None, 0):
            target = str(cred.get("TargetName") or "")
            if not target.endswith("token.com.apify.cli"):
                continue
            blob = cred.get("CredentialBlob")
            if not blob:
                continue
            if isinstance(blob, bytes):
                try:
                    return blob.decode("utf-16-le").strip()
                except UnicodeDecodeError:
                    return blob.decode("utf-8", errors="replace").strip()
            return str(blob).strip()
    except Exception:
        pass

    try:
        import keyring

        for service, user in (
            ("token.com.apify.cli", "token"),
            ("com.apify.cli", "token"),
            ("apify-cli", "token"),
        ):
            value = keyring.get_password(service, user)
            if value:
                return value.strip()
    except Exception:
        pass

    return None
