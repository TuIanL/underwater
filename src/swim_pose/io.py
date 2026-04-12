from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: str | Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    destination = ensure_parent(path)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: dict) -> None:
    destination = ensure_parent(path)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    destination = ensure_parent(path)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_toml(path: str | Path) -> dict:
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeError("TOML support requires Python 3.11+ or the 'tomli' package.") from exc
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)
