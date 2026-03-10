from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class PlayerStore:
    """
    Persistent JSON-backed player account store.
    Thread-safe for concurrent access from the FastAPI server and runtime thread.

    File structure:
        {
            "players": [
                {
                    "id": "uuid4",
                    "name": "Alice",
                    "created_at": "2026-03-03T00:00:00+00:00",
                    "career": {"total_shots": 0, "makes": 0, "fg_percent": 0.0},
                    "modes": {
                        "Spot Up": {"shots": 0, "makes": 0, "sessions": 0}
                    },
                    "sessions": [
                        {
                            "mode": "Spot Up",
                            "date": "2026-03-03T12:00:00+00:00",
                            "duration_sec": 240,
                            "total_shots": 20,
                            "makes": 12,
                            "fg_percent": 60.0
                        }
                    ]
                }
            ]
        }
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._write({"players": []})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_players(self) -> List[Dict]:
        players = list(self._read()["players"])
        players.sort(key=self._player_activity_ts, reverse=True)
        return players

    def get_player(self, player_id: str) -> Optional[Dict]:
        for p in self._read()["players"]:
            if p["id"] == player_id:
                return p
        return None

    def create_player(self, name: str) -> Dict:
        with self._lock:
            data = self._read()
            now_iso = datetime.now(timezone.utc).isoformat()
            player: Dict[str, Any] = {
                "id": str(uuid.uuid4()),
                "name": name.strip(),
                "created_at": now_iso,
                "last_active_at": now_iso,
                "career": {"total_shots": 0, "makes": 0, "fg_percent": 0.0},
                "modes": {},
                "sessions": [],
            }
            data["players"].append(player)
            self._write(data)
            return player

    def touch_player(self, player_id: str) -> Optional[Dict]:
        """Update the player's last active timestamp and return the updated player."""
        with self._lock:
            data = self._read()
            for p in data["players"]:
                if p["id"] != player_id:
                    continue
                p["last_active_at"] = datetime.now(timezone.utc).isoformat()
                self._write(data)
                return p
        return None

    def update_player_name(self, player_id: str, name: str) -> Optional[Dict]:
        with self._lock:
            data = self._read()
            for p in data["players"]:
                if p["id"] != player_id:
                    continue
                p["name"] = name.strip()
                p["last_active_at"] = datetime.now(timezone.utc).isoformat()
                self._write(data)
                return p
        return None

    def delete_player(self, player_id: str) -> bool:
        with self._lock:
            data = self._read()
            players = data.get("players", [])
            before = len(players)
            data["players"] = [p for p in players if p.get("id") != player_id]
            if len(data["players"]) == before:
                return False
            self._write(data)
            return True

    def add_session(self, player_id: str, session: Dict) -> bool:
        """
        Append a completed session and update career + per-mode aggregates.
        Returns True if the player was found and updated.
        """
        with self._lock:
            data = self._read()
            for p in data["players"]:
                if p["id"] != player_id:
                    continue
                p["sessions"].append(session)
                p["last_active_at"] = session.get("date") or datetime.now(timezone.utc).isoformat()

                total = int(session.get("total_shots", 0))
                makes = int(session.get("makes", 0))
                p["career"]["total_shots"] += total
                p["career"]["makes"] += makes
                ct = p["career"]["total_shots"]
                cm = p["career"]["makes"]
                p["career"]["fg_percent"] = round(cm / ct * 100.0, 1) if ct else 0.0

                mode_name = session.get("mode", "")
                if mode_name:
                    m = p["modes"].setdefault(
                        mode_name, {"shots": 0, "makes": 0, "sessions": 0}
                    )
                    m["shots"] += total
                    m["makes"] += makes
                    m["sessions"] += 1

                self._write(data)
                return True
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read(self) -> Dict:
        if not self._path.exists():
            return {"players": []}
        with self._path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, data: Dict) -> None:
        self._path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @staticmethod
    def _parse_ts(value: Any) -> float:
        if not value or not isinstance(value, str):
            return 0.0
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(raw).timestamp()
        except ValueError:
            return 0.0

    @classmethod
    def _player_activity_ts(cls, player: Dict[str, Any]) -> float:
        direct = cls._parse_ts(player.get("last_active_at"))
        if direct:
            return direct
        sessions = player.get("sessions") or []
        latest_session = 0.0
        for sess in sessions:
            latest_session = max(latest_session, cls._parse_ts((sess or {}).get("date")))
        if latest_session:
            return latest_session
        return cls._parse_ts(player.get("created_at"))
