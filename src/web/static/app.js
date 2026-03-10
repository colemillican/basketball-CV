/* ═══════════════════════════════════════════════════════════
   BASKETBALL TRAINER — App Logic
   Screens: player-select → add-player → mode-select → live → summary
   Input:   Xbox controller via Gamepad API + keyboard fallback
═══════════════════════════════════════════════════════════ */

"use strict";

// ════════════════════════════════════════════════════════════
// STATE
// ════════════════════════════════════════════════════════════

const App = {
  screen: "player-select",
  players: [],      // [{id, name, career, modes, sessions}]
  modes: [],        // [{name, end_trigger, timer_sec?}]
  currentPlayer: null,
  currentMode: null,
  historyFilter: "all", // all | last7 | mode
  historyRows: [],
  session: {
    makes: 0,
    misses: 0,
    total: 0,
    fg_percent: 0,
    running: false,
  },
  shots: [],        // [{shot_id, court_xy, result}] accumulated during live session
  navIndex: 0,      // focused item index in current list
};

const IDLE = {
  timeoutMs: 10 * 60 * 1000,
  timer: null,
  active: false,
};

function setIdleTimeoutFromBoot(boot) {
  const sec = Number(boot?.ui?.idle_timeout_sec);
  if (Number.isFinite(sec) && sec >= 30) {
    IDLE.timeoutMs = Math.floor(sec * 1000);
  }
}

function parseIsoTime(value) {
  if (!value || typeof value !== "string") return 0;
  const ts = Date.parse(value);
  return Number.isNaN(ts) ? 0 : ts;
}

function getPlayerActivityTs(player) {
  const direct = parseIsoTime(player?.last_active_at);
  if (direct) return direct;
  const sessions = Array.isArray(player?.sessions) ? player.sessions : [];
  let latestSession = 0;
  sessions.forEach((s) => {
    latestSession = Math.max(latestSession, parseIsoTime(s?.date));
  });
  if (latestSession) return latestSession;
  return parseIsoTime(player?.created_at);
}

function sortPlayersByRecentActivity(players) {
  if (!Array.isArray(players)) return [];
  return [...players].sort((a, b) => getPlayerActivityTs(b) - getPlayerActivityTs(a));
}

function sortAppPlayers() {
  App.players = sortPlayersByRecentActivity(App.players);
}

// ════════════════════════════════════════════════════════════
// API
// ════════════════════════════════════════════════════════════

async function api(path, method = "GET", body = null) {
  const res = await fetch(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : null,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text);
  }
  return res.json();
}

// ════════════════════════════════════════════════════════════
// SCREEN MANAGER
// ════════════════════════════════════════════════════════════

function showScreen(name) {
  document.querySelectorAll(".screen").forEach((s) => s.classList.remove("active"));
  const el = document.getElementById(`screen-${name}`);
  if (el) el.classList.add("active");
  App.screen = name;
  App.navIndex = 0;
  scheduleIdleTimer();
}

function clearIdleTimer() {
  if (IDLE.timer) {
    clearTimeout(IDLE.timer);
    IDLE.timer = null;
  }
}

function canEnterIdleMode() {
  return App.screen !== "live" && !App.session.running && App.screen !== "idle-leaderboard";
}

function scheduleIdleTimer() {
  clearIdleTimer();
  if (!canEnterIdleMode()) return;
  IDLE.timer = setTimeout(() => {
    enterIdleLeaderboard();
  }, IDLE.timeoutMs);
}

function renderIdleLeaderboard() {
  const list = document.getElementById("idle-leaderboard-list");
  if (!list) return;
  list.innerHTML = "";

  const sorted = [...App.players].sort((a, b) => {
    const makesA = Number(a?.career?.makes ?? 0);
    const makesB = Number(b?.career?.makes ?? 0);
    if (makesA !== makesB) return makesB - makesA;
    const shotsA = Number(a?.career?.total_shots ?? 0);
    const shotsB = Number(b?.career?.total_shots ?? 0);
    return shotsB - shotsA;
  }).slice(0, 8);

  if (sorted.length === 0) {
    const li = document.createElement("li");
    li.className = "nav-item";
    li.innerHTML = "<span>No players yet</span><span class=\"nav-sub\">Create one to begin tracking</span>";
    list.appendChild(li);
    return;
  }

  sorted.forEach((p, i) => {
    const makes = Number(p?.career?.makes ?? 0);
    const shots = Number(p?.career?.total_shots ?? 0);
    const fg = Number(p?.career?.fg_percent ?? 0);
    const li = document.createElement("li");
    li.className = "nav-item";
    li.innerHTML = `
      <span>#${i + 1} ${p.name}</span>
      <span class="nav-sub">${makes} makes · ${shots} shots · ${fg.toFixed(1)}% FG</span>
    `;
    list.appendChild(li);
  });
}

async function enterIdleLeaderboard() {
  if (!canEnterIdleMode()) return;
  IDLE.active = true;
  clearIdleTimer();
  try {
    const players = await api("/api/players");
    if (Array.isArray(players)) App.players = sortPlayersByRecentActivity(players);
  } catch (err) {
    console.error("Failed to refresh leaderboard players:", err);
  }
  renderIdleLeaderboard();
  showScreen("idle-leaderboard");
}

function exitIdleLeaderboard() {
  if (!IDLE.active) return;
  IDLE.active = false;
  App.currentPlayer = null;
  App.currentMode = null;
  App.historyFilter = "all";
  renderPlayerSelect();
  showScreen("player-select");
}

function noteUserActivity() {
  if (IDLE.active) {
    exitIdleLeaderboard();
    return true;
  }
  scheduleIdleTimer();
  return false;
}

// ════════════════════════════════════════════════════════════
// PLAYER SELECT SCREEN
// ════════════════════════════════════════════════════════════

function renderPlayerSelect() {
  const list = document.getElementById("player-list");
  list.innerHTML = "";
  App.players.forEach((p, i) => {
    const li = document.createElement("li");
    li.className = "nav-item";
    li.dataset.navIndex = i;
    const fg = p.career?.fg_percent ?? 0;
    const shots = p.career?.total_shots ?? 0;
    li.innerHTML = `
      <span>${p.name}</span>
      <span class="nav-sub">${shots} shots · ${fg.toFixed(1)}% FG</span>
    `;
    li.onclick = () => { App.navIndex = i; playerSelectConfirm(); };
    list.appendChild(li);
  });
  updatePlayerSelectFocus();
}

function updatePlayerSelectFocus() {
  const items = getPlayerSelectItems();
  items.forEach((el, i) => {
    el.classList.toggle("focused", i === App.navIndex);
  });
}

function getPlayerSelectItems() {
  const listItems = Array.from(document.querySelectorAll("#player-list .nav-item"));
  const addBtn = document.getElementById("btn-add-player");
  return [...listItems, addBtn];
}

function playerSelectConfirm() {
  const items = getPlayerSelectItems();
  const addBtn = document.getElementById("btn-add-player");
  const focused = items[App.navIndex];
  if (!focused) return;

  if (focused === addBtn) {
    enterAddPlayer();
    return;
  }

  const idx = parseInt(focused.dataset.navIndex, 10);
  App.currentPlayer = App.players[idx];
  if (App.currentPlayer?.id) {
    App.currentPlayer.last_active_at = new Date().toISOString();
    sortAppPlayers();
    const currentId = App.currentPlayer.id;
    void api(`/api/players/${currentId}/touch`, "POST")
      .then((updated) => {
        if (!updated?.id) return;
        App.players = App.players.map((p) => (p.id === updated.id ? updated : p));
        sortAppPlayers();
      })
      .catch((err) => {
        console.error("Failed to update player activity:", err);
      });
  }
  enterModeSelect();
}

// ════════════════════════════════════════════════════════════
// ADD PLAYER SCREEN (virtual keyboard)
// ════════════════════════════════════════════════════════════

const KB_ROWS = [
  ["A","B","C","D","E","F","G"],
  ["H","I","J","K","L","M","N"],
  ["O","P","Q","R","S","T","U"],
  ["V","W","X","Y","Z","⌫","✓"],
];
const KB_ROWS_COUNT = KB_ROWS.length;
const KB_COLS_COUNT = KB_ROWS[0].length;

let kbRow = 0;
let kbCol = 0;
let kbText = "";

function enterAddPlayer() {
  kbRow = 0; kbCol = 0; kbText = "";
  renderKeyboard();
  showScreen("add-player");
}

function renderKeyboard() {
  const container = document.getElementById("keyboard");
  container.innerHTML = "";
  KB_ROWS.forEach((row, ri) => {
    const rowEl = document.createElement("div");
    rowEl.className = "kb-row";
    row.forEach((key, ci) => {
      const btn = document.createElement("div");
      btn.className = "kb-key";
      if (key === "⌫" || key === "✓") btn.classList.add("key-special");
      if (key === "✓") btn.classList.add("key-confirm");
      if (ri === kbRow && ci === kbCol) btn.classList.add("focused");
      btn.textContent = key;
      btn.onclick = () => { kbRow = ri; kbCol = ci; kbTypeCurrentKey(); };
      rowEl.appendChild(btn);
    });
    container.appendChild(rowEl);
  });
  const display = document.getElementById("kb-display");
  display.textContent = kbText || "_";
}

function kbTypeCurrentKey() {
  const key = KB_ROWS[kbRow][kbCol];
  if (key === "⌫") {
    kbText = kbText.slice(0, -1);
  } else if (key === "✓") {
    submitNewPlayer();
    return;
  } else {
    if (kbText.length < 20) kbText += key;
  }
  renderKeyboard();
}

function kbBackspace() {
  kbText = kbText.slice(0, -1);
  renderKeyboard();
}

async function submitNewPlayer() {
  const name = kbText.trim();
  if (!name) return;
  try {
    const player = await api("/api/players", "POST", { name });
    App.players.push(player);
    sortAppPlayers();
    App.currentPlayer = player;
    enterModeSelect();
  } catch (err) {
    console.error("Failed to create player:", err);
  }
}

// ════════════════════════════════════════════════════════════
// MODE SELECT SCREEN
// ════════════════════════════════════════════════════════════

function enterModeSelect() {
  renderModeSelect();
  showScreen("mode-select");
}

function renderModeSelect() {
  // Player badge
  const badge = document.getElementById("mode-player-badge");
  badge.textContent = App.currentPlayer?.name?.toUpperCase() ?? "";

  // Career stats bar
  const career = App.currentPlayer?.career ?? {};
  const careerBar = document.getElementById("player-career-bar");
  const shots = career.total_shots ?? 0;
  const makes = career.makes ?? 0;
  const fg = career.fg_percent ?? 0;
  careerBar.textContent = shots > 0
    ? `Career: ${shots} shots · ${makes} makes · ${fg.toFixed(1)}% FG`
    : "No career stats yet";

  // Mode list
  const list = document.getElementById("mode-list");
  list.innerHTML = "";
  App.modes.forEach((m, i) => {
    const li = document.createElement("li");
    li.className = "nav-item";
    li.dataset.navIndex = i;
    const sub = m.end_trigger === "timer"
      ? `${Math.floor((m.timer_sec || 0) / 60)} min timed`
      : "Manual stop";
    li.innerHTML = `<span>${m.name}</span><span class="nav-sub">${sub}</span>`;
    li.onclick = () => { App.navIndex = i; modeSelectConfirm(); };
    list.appendChild(li);
  });

  const profileBtn = document.getElementById("btn-view-profile");
  if (profileBtn) {
    profileBtn.dataset.navIndex = String(App.modes.length);
    profileBtn.onclick = () => { App.navIndex = App.modes.length; modeSelectConfirm(); };
  }

  updateModeSelectFocus();
}

function updateModeSelectFocus() {
  const items = getModeSelectItems();
  items.forEach((el, i) => el.classList.toggle("focused", i === App.navIndex));
}

function getModeSelectItems() {
  const listItems = Array.from(document.querySelectorAll("#mode-list .nav-item"));
  const profileBtn = document.getElementById("btn-view-profile");
  return profileBtn ? [...listItems, profileBtn] : listItems;
}

function modeSelectConfirm() {
  const items = getModeSelectItems();
  const focused = items[App.navIndex];
  if (!focused) return;
  if (focused.id === "btn-view-profile") {
    enterProfile();
    return;
  }
  const idx = parseInt(focused.dataset.navIndex, 10);
  if (Number.isNaN(idx) || !App.modes[idx]) return;
  App.currentMode = App.modes[idx];
  startSession();
}

// ════════════════════════════════════════════════════════════
// SESSION START / LIVE SCREEN
// ════════════════════════════════════════════════════════════

let timerInterval = null;
let timerRemaining = 0;
let liveCourtChart = null;
let sessionStartTime = null;
let stopPromise = null;
let startInFlight = false;

async function startSession() {
  if (startInFlight) return;
  startInFlight = true;

  try {
    // Ensure prior stop/finalization completed before starting next session.
    if (stopPromise) await stopPromise;

    await api("/api/session/start", "POST", {
      player_id: App.currentPlayer?.id ?? "",
      player_name: App.currentPlayer?.name ?? "",
      mode: App.currentMode?.name ?? "",
    });
  } catch (err) {
    console.error("Failed to start session:", err);
    return;
  }

  try {
    App.shots = [];
    App.session = { makes: 0, misses: 0, total: 0, fg_percent: 0, running: true };
    sessionStartTime = Date.now();

    // Prepare live screen UI
    document.getElementById("live-player-name").textContent =
      (App.currentPlayer?.name ?? "").toUpperCase();
    document.getElementById("live-mode-name").textContent =
      (App.currentMode?.name ?? "").toUpperCase();
    updateLiveStats();
    setLastShotDisplay(null);

    // Setup timer
    const timerBlock = document.getElementById("timer-block");
    if (App.currentMode?.end_trigger === "timer" && App.currentMode?.timer_sec) {
      timerRemaining = App.currentMode.timer_sec;
      timerBlock.style.display = "";
      renderTimer();
      timerInterval = setInterval(() => {
        timerRemaining--;
        renderTimer();
        if (timerRemaining <= 0) endSession();
      }, 1000);
    } else {
      timerRemaining = 0;
      timerBlock.style.display = "none";
      document.getElementById("live-timer").textContent = "—";
    }

    // Build court chart
    const svg = document.getElementById("shot-chart");
    svg.innerHTML = "";
    liveCourtChart = new CourtChart(svg);

    showScreen("live");
  } finally {
    startInFlight = false;
  }
}

function renderTimer() {
  const mins = Math.floor(timerRemaining / 60);
  const secs = timerRemaining % 60;
  document.getElementById("live-timer").textContent =
    `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

async function endSession() {
  if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
  const shouldNavigate = App.screen === "live";
  App.session.running = false;

  // Navigate immediately for responsive UX while backend finalization completes.
  if (shouldNavigate) enterSummary();

  if (!stopPromise) {
    stopPromise = api("/api/session/stop", "POST")
      .catch((err) => {
        console.error("Failed to stop session:", err);
      })
      .finally(() => {
        stopPromise = null;
      });
  }
  await stopPromise;
}

function updateLiveStats() {
  const s = App.session;
  const makesEl  = document.getElementById("live-makes");
  const missesEl = document.getElementById("live-misses");
  makesEl.textContent  = String(s.makes);
  missesEl.textContent = String(s.misses);
  document.getElementById("live-fg").textContent = s.fg_percent.toFixed(1);
  const bar = document.getElementById("live-fg-bar");
  if (bar) bar.style.width = `${Math.min(100, s.fg_percent)}%`;
}

function setLastShotDisplay(result) {
  const el = document.getElementById("last-shot-result");
  el.classList.remove("make", "miss");
  if (result === "make") {
    el.textContent = "MAKE ✓";
    el.classList.add("make");
  } else if (result === "miss") {
    el.textContent = "MISS ✗";
    el.classList.add("miss");
  } else {
    el.textContent = "—";
  }
}

function triggerFlash(result) {
  const el = document.getElementById("shot-flash");
  el.classList.remove("flash-make", "flash-miss");
  void el.offsetWidth; // force reflow
  el.classList.add(result === "make" ? "flash-make" : "flash-miss");
}

function popStat(elId) {
  const el = document.getElementById(elId);
  if (!el) return;
  el.classList.remove("popping");
  void el.offsetWidth;
  el.classList.add("popping");
  el.addEventListener("animationend", () => el.classList.remove("popping"), { once: true });
}

function normalizeShot(shot) {
  if (!shot) return null;
  const court_xy = Array.isArray(shot.court_xy) && shot.court_xy.length === 2
    ? shot.court_xy
    : null;
  return {
    shot_id: shot.shot_id ?? null,
    court_xy,
    result: shot.result ?? null,
  };
}

function renderLiveShots() {
  if (!liveCourtChart) return;
  liveCourtChart.clear();
  App.shots.forEach(({ court_xy, result }) => liveCourtChart.addShot(court_xy, result));
}

function syncShotsFromState(state) {
  const incomingRaw = Array.isArray(state?.shots) ? state.shots : [];
  const incoming = incomingRaw.map(normalizeShot).filter(Boolean);
  if (incoming.length === 0 && App.shots.length === 0) return;

  const sameLength = incoming.length === App.shots.length;
  const sameIds = sameLength && incoming.every((s, i) => {
    const a = s.shot_id;
    const b = App.shots[i]?.shot_id;
    if (a == null || b == null) return false;
    return a === b;
  });
  if (sameLength && sameIds) return;

  App.shots = incoming;
  if (App.screen === "live") renderLiveShots();
}

// ════════════════════════════════════════════════════════════
// SESSION SUMMARY SCREEN
// ════════════════════════════════════════════════════════════

const SUMMARY_MENU = [
  { label: "PLAY AGAIN",   action: "play-again" },
  { label: "CHANGE MODE",  action: "change-mode" },
  { label: "LOG OUT",      action: "logout" },
];

function enterSummary() {
  const s = App.session;
  document.getElementById("summary-player").textContent =
    (App.currentPlayer?.name ?? "").toUpperCase();

  const modeName = App.currentMode?.name ?? "";
  const elapsedSec = sessionStartTime ? Math.floor((Date.now() - sessionStartTime) / 1000) : 0;
  const mins = Math.floor(elapsedSec / 60);
  const secs = elapsedSec % 60;
  document.getElementById("summary-meta").textContent =
    `${modeName} · ${mins}:${String(secs).padStart(2, "0")}`;

  document.getElementById("sum-makes").textContent  = String(s.makes);
  document.getElementById("sum-misses").textContent = String(s.misses);
  document.getElementById("sum-fg").textContent = `${s.fg_percent.toFixed(1)}%`;

  // Rebuild summary chart from accumulated shots
  const sumSvg = document.getElementById("summary-chart");
  sumSvg.innerHTML = "";
  const sumChart = new CourtChart(sumSvg);
  App.shots.forEach(({ court_xy, result }) => sumChart.addShot(court_xy, result));

  // Render summary menu
  const menu = document.getElementById("summary-menu");
  menu.innerHTML = "";
  SUMMARY_MENU.forEach((item, i) => {
    const li = document.createElement("li");
    li.className = "nav-item";
    li.dataset.navIndex = i;
    li.dataset.action = item.action;
    li.textContent = item.label;
    li.onclick = () => { App.navIndex = i; summaryConfirm(); };
    menu.appendChild(li);
  });

  showScreen("summary");
  // Update player data from server to reflect saved session
  refreshPlayers();
}

function updateSummaryFocus() {
  const items = Array.from(document.querySelectorAll("#summary-menu .nav-item"));
  items.forEach((el, i) => el.classList.toggle("focused", i === App.navIndex));
}

function summaryConfirm() {
  const items = Array.from(document.querySelectorAll("#summary-menu .nav-item"));
  const focused = items[App.navIndex];
  if (!focused) return;
  const action = focused.dataset.action;
  if (action === "play-again") {
    startSession();
  } else if (action === "change-mode") {
    enterModeSelect();
  } else if (action === "logout") {
    App.currentPlayer = null;
    App.currentMode = null;
    showScreen("player-select");
    renderPlayerSelect();
  }
}

async function refreshPlayers() {
  try {
    const players = await api("/api/players");
    App.players = sortPlayersByRecentActivity(players);
    // Update currentPlayer with fresh data
    if (App.currentPlayer) {
      const updated = App.players.find((p) => p.id === App.currentPlayer.id);
      if (updated) App.currentPlayer = updated;
    }
  } catch (_) {}
}

// PROFILE + HISTORY
const PROFILE_MENU = [
  { label: "SESSION HISTORY", action: "history" },
  { label: "EDIT NAME", action: "edit-name" },
  { label: "DELETE ACCOUNT", action: "delete-account" },
  { label: "BACK TO MODES", action: "back" },
];

const HISTORY_FILTERS = [
  { key: "all", label: "ALL SESSIONS" },
  { key: "last7", label: "LAST 7 DAYS" },
  { key: "mode", label: "CURRENT MODE" },
];

function formatUtcDate(isoString) {
  if (!isoString) return "Unknown date";
  const d = new Date(isoString);
  if (Number.isNaN(d.getTime())) return isoString;
  return d.toLocaleString();
}

function getSortedSessions(player) {
  return [...(player?.sessions ?? [])].sort((a, b) => {
    const ta = Date.parse(a?.date ?? 0) || 0;
    const tb = Date.parse(b?.date ?? 0) || 0;
    return tb - ta;
  });
}

async function loadCurrentPlayerDetails() {
  if (!App.currentPlayer?.id) return null;
  try {
    const p = await api(`/api/players/${App.currentPlayer.id}`);
    App.currentPlayer = p;
    return p;
  } catch (err) {
    console.error("Failed to load player profile:", err);
    return App.currentPlayer;
  }
}

async function enterProfile() {
  const player = await loadCurrentPlayerDetails();
  if (!player) return;
  App.navIndex = 0;

  document.getElementById("profile-player-badge").textContent = (player.name ?? "").toUpperCase();
  document.getElementById("profile-created-at").textContent = `Created ${formatUtcDate(player.created_at)}`;
  document.getElementById("profile-name-value").textContent = player.name ?? "";
  document.getElementById("profile-total-shots").textContent = String(player.career?.total_shots ?? 0);
  document.getElementById("profile-total-makes").textContent = String(player.career?.makes ?? 0);
  document.getElementById("profile-fg").textContent = `${Number(player.career?.fg_percent ?? 0).toFixed(1)}%`;

  const menu = document.getElementById("profile-menu");
  menu.innerHTML = "";
  PROFILE_MENU.forEach((item, i) => {
    const li = document.createElement("li");
    li.className = "nav-item";
    li.dataset.navIndex = i;
    li.dataset.action = item.action;
    li.textContent = item.label;
    li.onclick = () => { App.navIndex = i; profileConfirm(); };
    menu.appendChild(li);
  });

  showScreen("profile");
  updateProfileFocus();
}

function getProfileItems() {
  return Array.from(document.querySelectorAll("#profile-menu .nav-item"));
}

function updateProfileFocus() {
  const items = getProfileItems();
  items.forEach((el, i) => el.classList.toggle("focused", i === App.navIndex));
}

function profileConfirm() {
  const items = getProfileItems();
  const focused = items[App.navIndex];
  if (!focused) return;
  const action = focused.dataset.action;
  if (action === "history") {
    enterHistory();
  } else if (action === "edit-name") {
    editCurrentPlayerName();
  } else if (action === "delete-account") {
    deleteCurrentPlayer();
  } else {
    enterModeSelect();
  }
}

async function editCurrentPlayerName() {
  if (!App.currentPlayer?.id) return;
  const current = App.currentPlayer.name ?? "";
  const nextName = window.prompt("Enter a new player name:", current);
  if (nextName == null) return;
  const trimmed = nextName.trim();
  if (!trimmed) return;
  if (trimmed.length > 24) {
    window.alert("Name too long (max 24 chars).");
    return;
  }
  try {
    const updated = await api(`/api/players/${App.currentPlayer.id}`, "PUT", { name: trimmed });
    App.currentPlayer = updated;
    await refreshPlayers();
    await enterProfile();
  } catch (err) {
    console.error("Failed to rename player:", err);
    window.alert("Failed to rename player.");
  }
}

async function deleteCurrentPlayer() {
  if (!App.currentPlayer?.id) return;
  const name = App.currentPlayer.name ?? "this player";
  const ok = window.confirm(`Delete account "${name}"? This cannot be undone.`);
  if (!ok) return;
  try {
    await api(`/api/players/${App.currentPlayer.id}`, "DELETE");
    App.players = App.players.filter((p) => p.id !== App.currentPlayer.id);
    App.currentPlayer = null;
    App.currentMode = null;
    App.navIndex = 0;
    renderPlayerSelect();
    showScreen("player-select");
  } catch (err) {
    console.error("Failed to delete player:", err);
    window.alert("Failed to delete player.");
  }
}

function getFilteredHistorySessions(player) {
  const sessions = getSortedSessions(player);
  const now = Date.now();
  if (App.historyFilter === "last7") {
    const cutoff = now - 7 * 24 * 60 * 60 * 1000;
    return sessions.filter((s) => (Date.parse(s.date ?? 0) || 0) >= cutoff);
  }
  if (App.historyFilter === "mode") {
    const modeName = App.currentMode?.name;
    if (!modeName) return sessions;
    return sessions.filter((s) => s.mode === modeName);
  }
  return sessions;
}

function renderHistoryDetail(row) {
  const p = document.getElementById("history-detail-primary");
  const s = document.getElementById("history-detail-secondary");
  if (!row || row.type !== "session") {
    p.textContent = "Select a session";
    s.textContent = "Mode, date, and shooting summary appear here.";
    return;
  }
  const session = row.session;
  const misses = Math.max(0, Number(session.total_shots ?? 0) - Number(session.makes ?? 0));
  p.textContent = `${session.mode ?? "Mode"} · ${formatUtcDate(session.date)}`;
  s.textContent = `${session.total_shots ?? 0} shots · ${session.makes ?? 0} makes · ${misses} misses · ${Number(session.fg_percent ?? 0).toFixed(1)}% FG`;
}

function getHistoryItems() {
  return Array.from(document.querySelectorAll("#history-list .nav-item"));
}

function updateHistoryFocus() {
  const items = getHistoryItems();
  items.forEach((el, i) => el.classList.toggle("focused", i === App.navIndex));
  const row = App.historyRows[App.navIndex];
  renderHistoryDetail(row);
}

function renderHistory() {
  const player = App.currentPlayer;
  document.getElementById("history-player-badge").textContent = (player?.name ?? "").toUpperCase();
  const filter = HISTORY_FILTERS.find((f) => f.key === App.historyFilter) ?? HISTORY_FILTERS[0];
  document.getElementById("history-filter-label").textContent = filter.label;

  const sessions = getFilteredHistorySessions(player);
  const list = document.getElementById("history-list");
  list.innerHTML = "";
  App.historyRows = [];

  sessions.forEach((sess, i) => {
    const li = document.createElement("li");
    li.className = "nav-item";
    li.dataset.navIndex = i;
    const misses = Math.max(0, Number(sess.total_shots ?? 0) - Number(sess.makes ?? 0));
    li.innerHTML = `<span>${sess.mode ?? "Mode"} · ${Number(sess.fg_percent ?? 0).toFixed(1)}% FG</span><span class="nav-sub">${sess.makes ?? 0}/${sess.total_shots ?? 0} · ${misses} misses · ${formatUtcDate(sess.date)}</span>`;
    li.onclick = () => { App.navIndex = i; updateHistoryFocus(); };
    list.appendChild(li);
    App.historyRows.push({ type: "session", session: sess });
  });

  const backIndex = App.historyRows.length;
  const backLi = document.createElement("li");
  backLi.className = "nav-item";
  backLi.dataset.navIndex = backIndex;
  backLi.dataset.action = "back";
  backLi.textContent = "BACK TO PROFILE";
  backLi.onclick = () => { App.navIndex = backIndex; historyConfirm(); };
  list.appendChild(backLi);
  App.historyRows.push({ type: "back" });

  App.navIndex = clamp(App.navIndex, 0, App.historyRows.length - 1);
  updateHistoryFocus();
}

function cycleHistoryFilter(dir) {
  const idx = HISTORY_FILTERS.findIndex((f) => f.key === App.historyFilter);
  const next = (idx + dir + HISTORY_FILTERS.length) % HISTORY_FILTERS.length;
  App.historyFilter = HISTORY_FILTERS[next].key;
  App.navIndex = 0;
  renderHistory();
}

async function enterHistory() {
  await loadCurrentPlayerDetails();
  showScreen("history");
  App.navIndex = 0;
  renderHistory();
}

function historyConfirm() {
  const row = App.historyRows[App.navIndex];
  if (!row) return;
  if (row.type === "back") {
    enterProfile();
    return;
  }
  renderHistoryDetail(row);
}

// ════════════════════════════════════════════════════════════
// SVG COURT CHART
// ════════════════════════════════════════════════════════════

class CourtChart {
  constructor(svgEl) {
    this.svg = svgEl;
    this.S = 12;           // 12px per foot
    this.W = 50 * this.S;  // 600px
    this.H = 47 * this.S;  // 564px
    this._drawCourt();
    this._dotsGroup = this._el("g", {});
    this.svg.appendChild(this._dotsGroup);
  }

  // Court coords → SVG pixel coords
  cx(x) { return x * this.S; }
  cy(y) { return (47 - y) * this.S; }

  _el(tag, attrs) {
    const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
    return el;
  }

  _drawCourt() {
    const S = this.S;
    const stroke = "#1e3a5f";
    const strokeW = 1.5;

    // Court surface
    this.svg.appendChild(this._el("rect", {
      x: 0, y: 0, width: this.W, height: this.H,
      fill: "#0d1b2e", stroke: stroke, "stroke-width": 2,
    }));

    // Baseline label ticks (subtle)
    this.svg.appendChild(this._el("line", {
      x1: 0, y1: this.cy(0), x2: this.W, y2: this.cy(0),
      stroke: stroke, "stroke-width": strokeW,
    }));

    // Paint
    this.svg.appendChild(this._el("rect", {
      x: this.cx(17), y: this.cy(19),
      width: 16 * S, height: 19 * S,
      fill: "rgba(0,212,255,0.04)", stroke: stroke, "stroke-width": strokeW,
    }));

    // Backboard
    this.svg.appendChild(this._el("line", {
      x1: this.cx(22), y1: this.cy(4),
      x2: this.cx(28), y2: this.cy(4),
      stroke: "#94a3b8", "stroke-width": 3,
    }));

    // Hoop
    this.svg.appendChild(this._el("circle", {
      cx: this.cx(25), cy: this.cy(5.25), r: 0.75 * S,
      fill: "none", stroke: "#f97316", "stroke-width": 2.5,
    }));

    // Free-throw circle (top half only — arc above FT line)
    const ftCX = this.cx(25);
    const ftCY = this.cy(19);
    const ftR  = 6 * S;
    // Full circle but dashed below FT line
    this.svg.appendChild(this._el("circle", {
      cx: ftCX, cy: ftCY, r: ftR,
      fill: "none", stroke: stroke, "stroke-width": strokeW,
    }));

    // 3-point arc
    const arcR  = 23.75 * S;
    const baskX = this.cx(25);
    const baskY = this.cy(5.25);
    const a1    = 22 * Math.PI / 180;
    const a2    = 158 * Math.PI / 180;
    const sx = baskX + arcR * Math.cos(a1);
    const sy = baskY - arcR * Math.sin(a1);
    const ex = baskX + arcR * Math.cos(a2);
    const ey = baskY - arcR * Math.sin(a2);
    this.svg.appendChild(this._el("path", {
      d: `M ${sx} ${sy} A ${arcR} ${arcR} 0 0 0 ${ex} ${ey}`,
      fill: "none", stroke: stroke, "stroke-width": strokeW,
    }));

    // Corner 3 lines
    this.svg.appendChild(this._el("line", {
      x1: this.cx(3), y1: this.cy(0),
      x2: this.cx(3), y2: this.cy(14),
      stroke: stroke, "stroke-width": strokeW,
    }));
    this.svg.appendChild(this._el("line", {
      x1: this.cx(47), y1: this.cy(0),
      x2: this.cx(47), y2: this.cy(14),
      stroke: stroke, "stroke-width": strokeW,
    }));
  }

  addShot(court_xy, result) {
    if (!court_xy) return;
    const [x, y] = court_xy;
    const svgX = this.cx(x);
    const svgY = this.cy(y);
    const color = result === "make" ? "#00ff88" : "#ff4444";
    const dot = this._el("circle", {
      cx: svgX, cy: svgY, r: 7,
      fill: color,
      opacity: 0.85,
    });
    // filter must be set via CSS style, not as an SVG attribute
    dot.style.filter = result === "make"
      ? "drop-shadow(0 0 5px #00ff88)"
      : "drop-shadow(0 0 4px #ff4444)";
    this._dotsGroup.appendChild(dot);
  }

  clear() {
    while (this._dotsGroup.firstChild) {
      this._dotsGroup.removeChild(this._dotsGroup.firstChild);
    }
  }
}

// ════════════════════════════════════════════════════════════
// WEBSOCKET
// ════════════════════════════════════════════════════════════

let pollTimer = null;

function connectWebSocket() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  let reconnectQueued = false;

  ws.onopen = () => {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  };

  ws.onmessage = (evt) => {
    try {
      const data = JSON.parse(evt.data);
      handleServerEvent(data);
    } catch (_) {}
  };

  const reconnect = () => {
    if (reconnectQueued) return;
    reconnectQueued = true;

    if (!pollTimer) {
      pollTimer = setInterval(async () => {
        try {
          const st = await api("/api/state");
          handleServerState(st);
        } catch (_) {}
      }, 1500);
    }
    setTimeout(connectWebSocket, 3000);
  };

  ws.onerror = reconnect;
  ws.onclose = reconnect;
}

function handleServerEvent(data) {
  const event = data.event;
  const state = data.state;

  if (event?.type === "shot") {
    const result   = event.result;
    const shot = normalizeShot(event);
    App.session.makes      = event.makes ?? App.session.makes;
    App.session.misses     = (event.total ?? App.session.total) - (event.makes ?? App.session.makes);
    App.session.total      = event.total ?? App.session.total;
    App.session.fg_percent = event.fg_percent ?? App.session.fg_percent;

    // Only update UI if on live screen
    if (App.screen === "live") {
      const seen = shot?.shot_id != null && App.shots.some((s) => s.shot_id === shot.shot_id);
      if (!seen && shot) {
        App.shots.push(shot);
        if (liveCourtChart) liveCourtChart.addShot(shot.court_xy, result);
      }
      updateLiveStats();
      setLastShotDisplay(result);
      triggerFlash(result);
      popStat(result === "make" ? "live-makes" : "live-misses");
    }
  }

  if (event?.type === "session_complete" && App.screen === "live") {
    App.session.running = false;
    if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
    enterSummary();
  }

  if (state) handleServerState(state);
}

function handleServerState(state) {
  syncShotsFromState(state);

  // Sync live stats if session is running and we're on the live screen
  if (state.running && App.screen === "live") {
    App.session.makes      = state.makes ?? App.session.makes;
    App.session.misses     = state.misses ?? App.session.misses;
    App.session.total      = state.total ?? App.session.total;
    App.session.fg_percent = state.fg_percent ?? App.session.fg_percent;
    setLastShotDisplay(state.last_shot?.result ?? null);
    updateLiveStats();
  }
}

// ════════════════════════════════════════════════════════════
// GAMEPAD CONTROLLER
// ════════════════════════════════════════════════════════════

const GAMEPAD = {
  BTN_A:     0,
  BTN_B:     1,
  BTN_START: 9,
  DPAD_UP:   12,
  DPAD_DOWN: 13,
  DPAD_LEFT: 14,
  DPAD_RIGHT:15,
};

const gamepadState = {
  prev: {},
  repeatAt: {},
  REPEAT_INIT_MS: 400,
  REPEAT_CONT_MS: 140,
};

function pollGamepad() {
  const gamepads = navigator.getGamepads ? navigator.getGamepads() : [];
  const gp = Array.from(gamepads).find((g) => g !== null);

  if (gp) {
    const now = Date.now();
    gp.buttons.forEach((btn, i) => {
      const pressed = btn.pressed || btn.value > 0.5;
      const wasPrev = gamepadState.prev[i] || false;

      if (pressed && !wasPrev) {
        // Fresh press
        onButton(i);
        gamepadState.repeatAt[i] = now + gamepadState.REPEAT_INIT_MS;
      } else if (pressed && wasPrev) {
        // Auto-repeat for d-pad
        if ([GAMEPAD.DPAD_UP, GAMEPAD.DPAD_DOWN, GAMEPAD.DPAD_LEFT, GAMEPAD.DPAD_RIGHT].includes(i)) {
          if (now >= gamepadState.repeatAt[i]) {
            onButton(i);
            gamepadState.repeatAt[i] = now + gamepadState.REPEAT_CONT_MS;
          }
        }
      }
      gamepadState.prev[i] = pressed;
    });

    // Left stick as D-pad fallback
    const axisY = gp.axes[1] ?? 0;
    const axisX = gp.axes[0] ?? 0;
    const DEAD = 0.5;
    if (Math.abs(axisY) > DEAD || Math.abs(axisX) > DEAD) {
      const key = `axis`;
      if (now >= (gamepadState.repeatAt[key] || 0)) {
        if (axisY < -DEAD) onButton(GAMEPAD.DPAD_UP);
        else if (axisY > DEAD) onButton(GAMEPAD.DPAD_DOWN);
        else if (axisX < -DEAD) onButton(GAMEPAD.DPAD_LEFT);
        else if (axisX > DEAD) onButton(GAMEPAD.DPAD_RIGHT);
        gamepadState.repeatAt[key] = now + gamepadState.REPEAT_CONT_MS;
      }
    } else {
      delete gamepadState.repeatAt[`axis`];
    }
  }

  requestAnimationFrame(pollGamepad);
}

function onButton(btn) {
  const wokeFromIdle = noteUserActivity();
  if (wokeFromIdle) return;

  const scr = App.screen;

  if (btn === GAMEPAD.DPAD_UP)    navigate(-1);
  if (btn === GAMEPAD.DPAD_DOWN)  navigate(1);
  if (btn === GAMEPAD.DPAD_LEFT)  navigateH(-1);
  if (btn === GAMEPAD.DPAD_RIGHT) navigateH(1);

  if (btn === GAMEPAD.BTN_A) {
    if (scr === "player-select") playerSelectConfirm();
    else if (scr === "add-player")  kbTypeCurrentKey();
    else if (scr === "mode-select") modeSelectConfirm();
    else if (scr === "profile")     profileConfirm();
    else if (scr === "history")     historyConfirm();
    else if (scr === "live")        document.getElementById("btn-end-session").classList.toggle("focused");
    else if (scr === "summary")     summaryConfirm();
  }

  if (btn === GAMEPAD.BTN_B) {
    if (scr === "add-player")  kbBackspace();
    else if (scr === "mode-select") {
      App.currentPlayer = null;
      showScreen("player-select");
      renderPlayerSelect();
    } else if (scr === "profile") {
      enterModeSelect();
    } else if (scr === "history") {
      enterProfile();
    } else if (scr === "summary") {
      startSession(); // Play again
    }
  }

  if (btn === GAMEPAD.BTN_START) {
    if (scr === "live") endSession();
    else if (scr === "add-player") submitNewPlayer();
  }
}

function navigate(dir) {
  const scr = App.screen;
  if (scr === "player-select") {
    const items = getPlayerSelectItems();
    App.navIndex = clamp(App.navIndex + dir, 0, items.length - 1);
    updatePlayerSelectFocus();
  } else if (scr === "mode-select") {
    const items = getModeSelectItems();
    App.navIndex = clamp(App.navIndex + dir, 0, items.length - 1);
    updateModeSelectFocus();
  } else if (scr === "add-player") {
    kbRow = clamp(kbRow + dir, 0, KB_ROWS_COUNT - 1);
    renderKeyboard();
  } else if (scr === "profile") {
    const items = getProfileItems();
    App.navIndex = clamp(App.navIndex + dir, 0, items.length - 1);
    updateProfileFocus();
  } else if (scr === "history") {
    const items = getHistoryItems();
    App.navIndex = clamp(App.navIndex + dir, 0, items.length - 1);
    updateHistoryFocus();
  } else if (scr === "summary") {
    App.navIndex = clamp(App.navIndex + dir, 0, SUMMARY_MENU.length - 1);
    updateSummaryFocus();
  }
}

function navigateH(dir) {
  if (App.screen === "add-player") {
    kbCol = clamp(kbCol + dir, 0, KB_ROWS[kbRow].length - 1);
    renderKeyboard();
  } else if (App.screen === "history") {
    cycleHistoryFilter(dir);
  }
}

function clamp(val, min, max) {
  return Math.max(min, Math.min(max, val));
}

// ════════════════════════════════════════════════════════════
// KEYBOARD FALLBACK (for development without controller)
// ════════════════════════════════════════════════════════════

document.addEventListener("keydown", (e) => {
  const wokeFromIdle = noteUserActivity();
  if (wokeFromIdle) {
    e.preventDefault();
    return;
  }

  const map = {
    ArrowUp:    GAMEPAD.DPAD_UP,
    ArrowDown:  GAMEPAD.DPAD_DOWN,
    ArrowLeft:  GAMEPAD.DPAD_LEFT,
    ArrowRight: GAMEPAD.DPAD_RIGHT,
    Enter:      GAMEPAD.BTN_A,
    Backspace:  GAMEPAD.BTN_B,
    Escape:     GAMEPAD.BTN_START,
  };
  if (map[e.key] !== undefined) {
    e.preventDefault();
    onButton(map[e.key]);
  }
}, { capture: true });

["mousemove", "mousedown", "touchstart", "wheel"].forEach((type) => {
  document.addEventListener(type, () => {
    noteUserActivity();
  }, { passive: true });
});

// Hover → focus for mouse users
document.addEventListener("mouseover", (e) => {
  noteUserActivity();
  const item = e.target.closest(".nav-item, .mode-profile-btn");
  if (!item) return;
  let idx = parseInt(item.dataset.navIndex, 10);
  if (item.id === "btn-view-profile") idx = App.modes.length;
  if (!isNaN(idx) && idx !== App.navIndex) {
    App.navIndex = idx;
    if (App.screen === "player-select") updatePlayerSelectFocus();
    else if (App.screen === "mode-select") updateModeSelectFocus();
    else if (App.screen === "profile") updateProfileFocus();
    else if (App.screen === "history") updateHistoryFocus();
    else if (App.screen === "summary") updateSummaryFocus();
  }
});

// ════════════════════════════════════════════════════════════
// INIT
// ════════════════════════════════════════════════════════════

async function init() {
  try {
    const boot = await api("/api/bootstrap");
    setIdleTimeoutFromBoot(boot);
    App.players = sortPlayersByRecentActivity(boot.players ?? []);
    App.modes   = boot.modes   ?? [];

    // If a session was already running when we loaded (e.g. page refresh), handle it
    const state = boot.state ?? {};
    if (state.running) {
      // Reconnect to live screen
      App.currentPlayer = App.players.find((p) => p.id === state.player_id) ?? null;
      App.currentMode   = App.modes.find((m) => m.name === state.mode) ?? null;
      App.session.makes      = state.makes ?? 0;
      App.session.misses     = state.misses ?? 0;
      App.session.total      = state.total ?? 0;
      App.session.fg_percent = state.fg_percent ?? 0;
      App.session.running    = true;
      App.shots              = (state.shots ?? []).map(normalizeShot).filter(Boolean);

      document.getElementById("live-player-name").textContent =
        (App.currentPlayer?.name ?? state.player ?? "").toUpperCase();
      document.getElementById("live-mode-name").textContent =
        (state.mode ?? "").toUpperCase();
      updateLiveStats();
      setLastShotDisplay(state.last_shot?.result ?? null);

      const svg = document.getElementById("shot-chart");
      svg.innerHTML = "";
      liveCourtChart = new CourtChart(svg);
      renderLiveShots();
      showScreen("live");
    } else {
      renderPlayerSelect();
      showScreen("player-select");
    }
  } catch (err) {
    console.error("Failed to initialise:", err);
  }

  // Static button click handlers
  document.getElementById("btn-add-player").onclick = enterAddPlayer;
  document.getElementById("btn-end-session").onclick = endSession;
  document.getElementById("btn-view-profile").onclick = () => {
    App.navIndex = App.modes.length;
    modeSelectConfirm();
  };

  scheduleIdleTimer();
  connectWebSocket();
  requestAnimationFrame(pollGamepad);
}

init();
