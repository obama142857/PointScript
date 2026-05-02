import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_PROJECT_DIRS = {
    "pcot_dir_name": "PcotPoints",
    "fact_dir_name": "FactPoints",
}
PROJECT_DIR_CONFIG_FILE = "automation_project_dirs.json"


def load_project_dirs(repo_root: Path) -> Dict[str, str]:
    cfg_path = repo_root / PROJECT_DIR_CONFIG_FILE
    cfg_data: Dict[str, Any] = {}
    if cfg_path.exists() and cfg_path.is_file():
        try:
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                cfg_data = raw
        except Exception:
            cfg_data = {}

    return {
        "pcot_dir_name": _pick_dir_name(
            cfg_data,
            keys=("pcot_dir_name", "pcot_project_dirname", "pcot_project_dir"),
            default=DEFAULT_PROJECT_DIRS["pcot_dir_name"],
        ),
        "fact_dir_name": _pick_dir_name(
            cfg_data,
            keys=("fact_dir_name", "fact_project_dirname", "fact_project_dir"),
            default=DEFAULT_PROJECT_DIRS["fact_dir_name"],
        ),
    }


def _pick_dir_name(cfg_data: Dict[str, Any], keys: tuple[str, ...], default: str) -> str:
    for key in keys:
        value = cfg_data.get(key)
        if isinstance(value, str):
            value = value.strip()
            if value:
                return value
    return default
