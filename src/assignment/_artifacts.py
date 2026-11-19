"""Write JSON artifacts without exposing partially written files."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile


def write_json_artifact(path: str | Path, data: object) -> None:
    destination = Path(path)
    serialized = json.dumps(data, ensure_ascii=False, indent=2)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary = NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            temporary.write(serialized)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
