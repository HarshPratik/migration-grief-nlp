"""Shared post-ID registry for cross-run deduplication."""

from __future__ import annotations

import json
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "data" / "raw" / "seen_post_ids.json"


def normalize_post_id(item: dict) -> str | None:
    pid = item.get("parsedId") or item.get("id")
    if not pid:
        return None
    return str(pid).removeprefix("t3_").strip() or None


def load_ids_from_dumps(raw_root: Path | None = None) -> set[str]:
    raw_root = raw_root or (ROOT / "data" / "raw")
    seen: set[str] = set()
    for path in raw_root.rglob("*.json"):
        if path.name.startswith("scrape_log") or path.name == "seen_post_ids.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        items = data.get("items") if isinstance(data, dict) else data
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            pid = normalize_post_id(it)
            if pid:
                seen.add(pid)
    return seen


class SeenIdRegistry:
    """Thread-safe post ID set with optional disk persistence."""

    def __init__(self, path: Path = DEFAULT_REGISTRY, seed_from_dumps: bool = True):
        self.path = path
        self._lock = threading.Lock()
        self.ids: set[str] = set()
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.ids = set(data.get("ids") or [])
            except (json.JSONDecodeError, OSError):
                self.ids = set()
        if seed_from_dumps:
            self.ids |= load_ids_from_dumps()

    def __len__(self) -> int:
        with self._lock:
            return len(self.ids)

    def filter_new(self, items: list[dict]) -> tuple[list[dict], int, int]:
        """Return (new_items, n_new, n_dup). Registers new IDs immediately."""
        new_items: list[dict] = []
        n_dup = 0
        with self._lock:
            for it in items:
                pid = normalize_post_id(it)
                if not pid:
                    # keep items without id (rare) but don't register
                    new_items.append(it)
                    continue
                if pid in self.ids:
                    n_dup += 1
                    continue
                self.ids.add(pid)
                new_items.append(it)
        return new_items, len(new_items), n_dup

    def save(self) -> None:
        with self._lock:
            ids = sorted(self.ids)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"count": len(ids), "ids": ids}, indent=2),
            encoding="utf-8",
        )
