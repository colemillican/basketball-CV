from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class AppConfig:
    raw: Dict[str, Any]

    @property
    def video(self) -> Dict[str, Any]:
        return self.raw["video"]

    @property
    def tracking(self) -> Dict[str, Any]:
        return self.raw["tracking"]

    @property
    def shot_logic(self) -> Dict[str, Any]:
        return self.raw["shot_logic"]

    @property
    def calibration(self) -> Dict[str, Any]:
        return self.raw["calibration"]

    @property
    def output(self) -> Dict[str, Any]:
        return self.raw["output"]


def load_config(path: str) -> AppConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig(raw=raw)
