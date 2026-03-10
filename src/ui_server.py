from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from config import load_config
from player_store import PlayerStore
from runtime import WorkoutRuntime


class StartRequest(BaseModel):
    player_id: str
    player_name: str
    mode: str


class CreatePlayerRequest(BaseModel):
    name: str

class UpdatePlayerRequest(BaseModel):
    name: str


class ConnectionManager:
    def __init__(self) -> None:
        self.connections: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.connections.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self.connections.discard(ws)

    async def broadcast(self, payload: Dict) -> None:
        dead: List[WebSocket] = []
        for ws in self.connections:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


class AppState:
    def __init__(self, config_path: str) -> None:
        self.cfg = load_config(config_path)
        self.output_root = Path(self.cfg.output["session_dir"]).resolve()
        self.lock = threading.Lock()
        self.manager = ConnectionManager()
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        ui_cfg = self.cfg.raw.get("ui", {})
        store_path = ui_cfg.get("player_store_path", "output/players.json")
        self.player_store = PlayerStore(store_path)
        self.idle_timeout_sec = int(ui_cfg.get("idle_timeout_sec", 600))

        # Mode objects loaded from config
        raw_modes = ui_cfg.get("modes", [])
        if raw_modes and isinstance(raw_modes[0], str):
            # Backwards compat: plain string list → objects with manual end_trigger
            self.modes = [{"name": m, "end_trigger": "manual"} for m in raw_modes]
        else:
            self.modes = raw_modes

        # Per-session tracking
        self._session_start_time: float = 0.0
        self._current_player_id: str = ""

        self.session: Dict = {
            "running": False,
            "player": "",
            "player_id": "",
            "mode": "",
            "makes": 0,
            "total": 0,
            "fg_percent": 0.0,
            "last_shot": None,
            "shots": [],
            "messages": [],
            "output": None,
        }

        self.runtime = WorkoutRuntime(config_path=config_path, on_event=self._on_runtime_event)

    def snapshot(self) -> Dict:
        with self.lock:
            makes = self.session["makes"]
            total = self.session["total"]
            return {
                "running": self.session["running"],
                "player": self.session["player"],
                "player_id": self.session["player_id"],
                "mode": self.session["mode"],
                "makes": makes,
                "misses": max(0, total - makes),
                "total": total,
                "fg_percent": self.session["fg_percent"],
                "last_shot": self.session["last_shot"],
                "shots": list(self.session["shots"]),
                "messages": self.session["messages"][-8:],
                "output": self.session["output"],
            }

    def _on_runtime_event(self, payload: Dict) -> None:
        if self.loop is None:
            return
        self.loop.call_soon_threadsafe(self.event_queue.put_nowait, payload)


_DEFAULT_CONFIG = str(Path(__file__).parent.parent / "configs" / "default.yaml")


def create_app(config_path: str = _DEFAULT_CONFIG) -> FastAPI:
    base_dir = Path(__file__).parent
    templates = Jinja2Templates(directory=str(base_dir / "web" / "templates"))
    output_root = Path(load_config(config_path).output["session_dir"]).resolve()

    app = FastAPI(title="Basketball CV")
    app.mount("/static", StaticFiles(directory=str(base_dir / "web" / "static")), name="static")
    app.mount("/session_output", StaticFiles(directory=str(output_root), check_dir=False), name="session_output")

    state = AppState(config_path=config_path)

    @app.on_event("startup")
    async def startup_event() -> None:
        state.loop = asyncio.get_running_loop()

        async def dispatcher() -> None:
            while True:
                event = await state.event_queue.get()
                with state.lock:
                    event_type = event.get("type")

                    if event_type == "status":
                        state.session["running"] = bool(event.get("running", state.session["running"]))
                        state.session["makes"] = int(event.get("makes", state.session["makes"]))
                        state.session["total"] = int(event.get("total", state.session["total"]))
                        state.session["fg_percent"] = float(event.get("fg_percent", state.session["fg_percent"]))

                    elif event_type == "shot":
                        shot_entry = {
                            "shot_id": event.get("shot_id"),
                            "result": event.get("result"),
                            "start_px": event.get("start_px"),
                            "court_xy": event.get("court_xy"),
                        }
                        state.session["last_shot"] = shot_entry
                        state.session["shots"].append(shot_entry)
                        state.session["makes"] = int(event.get("makes", state.session["makes"]))
                        state.session["total"] = int(event.get("total", state.session["total"]))
                        state.session["fg_percent"] = float(event.get("fg_percent", state.session["fg_percent"]))

                    elif event_type in ("warning", "error"):
                        state.session["messages"].append(
                            f"{event_type.upper()}: {event.get('message', '')}"
                        )

                    elif event_type == "session_complete":
                        state.session["running"] = False
                        output = event.get("output") or {}

                        # Build URL for shot chart
                        chart_url = None
                        chart_path = output.get("shot_chart")
                        if chart_path:
                            try:
                                rel = Path(chart_path).resolve().relative_to(state.output_root)
                                chart_url = f"/session_output/{rel.as_posix()}"
                            except Exception:
                                chart_url = None

                        state.session["output"] = {**output, "shot_chart_url": chart_url}

                        # Save completed session to player account
                        player_id = state._current_player_id
                        if player_id:
                            duration = int(time.time() - state._session_start_time)
                            state.player_store.add_session(player_id, {
                                "mode": state.session["mode"],
                                "date": datetime.now(timezone.utc).isoformat(),
                                "duration_sec": max(0, duration),
                                "total_shots": state.session["total"],
                                "makes": state.session["makes"],
                                "fg_percent": state.session["fg_percent"],
                            })

                await state.manager.broadcast({"type": "event", "event": event, "state": state.snapshot()})

        asyncio.create_task(dispatcher())

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    # ------------------------------------------------------------------
    # Player API
    # ------------------------------------------------------------------

    @app.get("/api/players")
    async def get_players() -> List[Dict]:
        return state.player_store.list_players()

    @app.post("/api/players")
    async def create_player(req: CreatePlayerRequest) -> Dict:
        name = req.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="Name cannot be empty")
        if len(name) > 24:
            raise HTTPException(status_code=400, detail="Name too long (max 24 chars)")
        return state.player_store.create_player(name)

    @app.get("/api/players/{player_id}")
    async def get_player(player_id: str) -> Dict:
        player = state.player_store.get_player(player_id)
        if player is None:
            raise HTTPException(status_code=404, detail="Player not found")
        return player

    @app.post("/api/players/{player_id}/touch")
    async def touch_player(player_id: str) -> Dict:
        player = state.player_store.touch_player(player_id)
        if player is None:
            raise HTTPException(status_code=404, detail="Player not found")
        return player

    @app.put("/api/players/{player_id}")
    async def update_player(player_id: str, req: UpdatePlayerRequest) -> Dict:
        name = req.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="Name cannot be empty")
        if len(name) > 24:
            raise HTTPException(status_code=400, detail="Name too long (max 24 chars)")
        player = state.player_store.update_player_name(player_id, name)
        if player is None:
            raise HTTPException(status_code=404, detail="Player not found")
        return player

    @app.delete("/api/players/{player_id}")
    async def delete_player(player_id: str) -> Dict:
        deleted = state.player_store.delete_player(player_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Player not found")
        if state.runtime.running and state.session.get("player_id") == player_id:
            state.runtime.stop()
        with state.lock:
            if state.session.get("player_id") == player_id:
                state.session["running"] = False
                state.session["player"] = ""
                state.session["player_id"] = ""
                state.session["mode"] = ""
                state.session["makes"] = 0
                state.session["total"] = 0
                state.session["fg_percent"] = 0.0
                state.session["last_shot"] = None
                state.session["shots"] = []
                state.session["messages"] = []
                state.session["output"] = None
                state._current_player_id = ""
        return {"ok": True}

    # ------------------------------------------------------------------
    # Bootstrap + state
    # ------------------------------------------------------------------

    @app.get("/api/bootstrap")
    async def bootstrap() -> Dict:
        return {
            "players": state.player_store.list_players(),
            "modes": state.modes,
            "state": state.snapshot(),
            "ui": {
                "idle_timeout_sec": state.idle_timeout_sec,
            },
        }

    @app.get("/api/state")
    async def get_state() -> Dict:
        return state.snapshot()

    # ------------------------------------------------------------------
    # Session control
    # ------------------------------------------------------------------

    @app.post("/api/session/start")
    async def start_session(req: StartRequest) -> Dict:
        if state.runtime.running:
            raise HTTPException(status_code=409, detail="Session already running")

        with state.lock:
            state.session["running"] = True
            state.session["player"] = req.player_name
            state.session["player_id"] = req.player_id
            state.session["mode"] = req.mode
            state.session["makes"] = 0
            state.session["total"] = 0
            state.session["fg_percent"] = 0.0
            state.session["last_shot"] = None
            state.session["shots"] = []
            state.session["messages"] = []
            state.session["output"] = None
            state._current_player_id = req.player_id
            state._session_start_time = time.time()

        state.runtime.start()
        await state.manager.broadcast({"type": "state", "state": state.snapshot()})
        return {"ok": True}

    @app.post("/api/session/stop")
    async def stop_session() -> Dict:
        state.runtime.stop()
        with state.lock:
            state.session["running"] = False
        await state.manager.broadcast({"type": "state", "state": state.snapshot()})
        return {"ok": True}

    # ------------------------------------------------------------------
    # WebSocket
    # ------------------------------------------------------------------

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await state.manager.connect(ws)
        try:
            await ws.send_json({"type": "state", "state": state.snapshot()})
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            state.manager.disconnect(ws)
        except Exception:
            state.manager.disconnect(ws)

    return app


app = create_app()
