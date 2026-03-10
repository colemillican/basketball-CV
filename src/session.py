import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class ShotRecord:
    shot_id: int
    frame_start: int
    frame_end: int
    result: str
    start_px_x: int
    start_px_y: int
    court_x_ft: Optional[float]
    court_y_ft: Optional[float]


class SessionStore:
    def __init__(self, output_dir: str) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(output_dir) / ts
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.records: List[ShotRecord] = []

    def add_shot(
        self,
        shot_id: int,
        frame_start: int,
        frame_end: int,
        result: str,
        start_px: Tuple[int, int],
        court_xy: Optional[Tuple[float, float]],
    ) -> None:
        cx = None if court_xy is None else round(court_xy[0], 2)
        cy = None if court_xy is None else round(court_xy[1], 2)
        self.records.append(
            ShotRecord(
                shot_id=shot_id,
                frame_start=frame_start,
                frame_end=frame_end,
                result=result,
                start_px_x=start_px[0],
                start_px_y=start_px[1],
                court_x_ft=cx,
                court_y_ft=cy,
            )
        )

    def save_csv(self, filename: str = "shots.csv") -> Path:
        path = self.session_dir / filename
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(self.records[0]).keys()) if self.records else [
                "shot_id", "frame_start", "frame_end", "result", "start_px_x", "start_px_y", "court_x_ft", "court_y_ft"
            ])
            w.writeheader()
            for r in self.records:
                w.writerow(asdict(r))
        return path

    def save_summary(self, filename: str = "summary.json") -> Path:
        makes = sum(1 for r in self.records if r.result == "make")
        total = len(self.records)
        summary = {
            "total_shots": total,
            "makes": makes,
            "misses": total - makes,
            "fg_percent": round((makes / total * 100.0), 1) if total else 0.0,
        }
        path = self.session_dir / filename
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return path
