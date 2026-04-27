import json
from pathlib import Path
from typing import Dict, Any
import re


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def make_template_name_from_file(file_path: Path) -> str:
    stem = file_path.stem.strip()

    match = re.search(r"(\d+)", stem)
    if match:
        number = match.group(1)
        return f"part_{number}"

    raise ValueError(
        f"Failed to extract detail number from file name: {file_path.name}"
    )
