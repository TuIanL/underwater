from __future__ import annotations

import json
import mimetypes
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .annotations import validate_annotation
from .constants import FRAME_STATUSES, KEYPOINT_NAMES, SKELETON_EDGES
from .io import read_json, write_json


HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Swim Pose Annotation</title>
  <style>
    :root {
      --bg: #0f172a;
      --panel: #111827;
      --panel-2: #1f2937;
      --line: #334155;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --accent: #22c55e;
      --warn: #f59e0b;
      --danger: #ef4444;
      --active: #38bdf8;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "PingFang SC", "Noto Sans SC", sans-serif;
      background: linear-gradient(180deg, #020617, #111827 38%, #0f172a);
      color: var(--text);
      min-height: 100vh;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(720px, 1fr) 360px;
      gap: 16px;
      padding: 16px;
      min-height: 100vh;
    }
    .viewer, .sidebar {
      background: rgba(17, 24, 39, 0.88);
      border: 1px solid rgba(148, 163, 184, 0.18);
      border-radius: 18px;
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.35);
      overflow: hidden;
    }
    .viewer-header, .sidebar-header {
      padding: 14px 16px;
      border-bottom: 1px solid rgba(148, 163, 184, 0.12);
      background: rgba(15, 23, 42, 0.72);
    }
    .viewer-header h1, .sidebar-header h2 {
      margin: 0;
      font-size: 18px;
    }
    .meta, .hint, .status-text {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
      white-space: pre-wrap;
    }
    .canvas-wrap {
      padding: 12px;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: calc(100vh - 170px);
    }
    canvas {
      max-width: 100%;
      max-height: calc(100vh - 210px);
      background: #000;
      border-radius: 12px;
      border: 1px solid rgba(148, 163, 184, 0.12);
      cursor: crosshair;
    }
    .sidebar {
      display: grid;
      grid-template-rows: auto auto auto 1fr auto;
    }
    .toolbar, .status-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      padding: 12px 16px 0;
    }
    .status-grid {
      grid-template-columns: repeat(2, 1fr);
      padding-top: 12px;
    }
    button {
      border: 0;
      border-radius: 12px;
      padding: 10px 12px;
      background: var(--panel-2);
      color: var(--text);
      font-weight: 600;
      cursor: pointer;
      transition: transform 120ms ease, background 120ms ease;
    }
    button:hover { transform: translateY(-1px); }
    button.primary { background: #2563eb; }
    button.good { background: #15803d; }
    button.warn { background: #b45309; }
    button.danger { background: #b91c1c; }
    .list {
      overflow: auto;
      padding: 12px 16px 16px;
    }
    .kp {
      display: grid;
      grid-template-columns: 20px 1fr auto;
      gap: 10px;
      align-items: center;
      padding: 10px 12px;
      margin-bottom: 8px;
      border-radius: 12px;
      background: rgba(31, 41, 55, 0.75);
      border: 1px solid transparent;
      cursor: pointer;
    }
    .kp.active {
      border-color: rgba(56, 189, 248, 0.8);
      background: rgba(14, 116, 144, 0.28);
    }
    .kp small { color: var(--muted); }
    .footer {
      padding: 12px 16px 16px;
      border-top: 1px solid rgba(148, 163, 184, 0.12);
    }
    .badge {
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      margin-top: 8px;
      background: rgba(56, 189, 248, 0.16);
      color: #7dd3fc;
    }
    .topline {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
    }
    @media (max-width: 1180px) {
      .layout {
        grid-template-columns: 1fr;
      }
      .canvas-wrap {
        min-height: auto;
      }
      canvas {
        max-height: 70vh;
      }
    }
  </style>
</head>
<body>
  <div class="layout">
    <section class="viewer">
      <div class="viewer-header">
        <div class="topline">
          <h1>人工标注界面</h1>
          <span id="fileCounter" class="badge">0 / 0</span>
        </div>
        <div id="meta" class="meta">正在加载...</div>
      </div>
      <div class="canvas-wrap">
        <canvas id="canvas"></canvas>
      </div>
    </section>

    <aside class="sidebar">
      <div class="sidebar-header">
        <h2>关键点</h2>
        <div id="statusText" class="status-text"></div>
      </div>

      <div class="toolbar">
        <button id="prevBtn">上一张 (A)</button>
        <button id="saveBtn" class="primary">保存 (Ctrl+S)</button>
        <button id="nextBtn">下一张 (D)</button>
      </div>

      <div class="toolbar">
        <button id="undoBtn">撤销 (Ctrl+Z)</button>
        <button id="clearBtn" class="warn">清当前点</button>
        <button id="clearFrameBtn" class="danger">清整帧</button>
      </div>

      <div class="status-grid">
        <button id="labeledBtn" class="good">Labeled (L)</button>
        <button id="noSwimmerBtn" class="warn">No Swimmer (N)</button>
        <button id="reviewBtn">Review (R)</button>
        <button id="pendingBtn">Pending (P)</button>
      </div>

      <div id="keypointList" class="list"></div>

      <div class="footer">
        <div class="hint">
左键打点
右键 / Delete / Backspace / C 删除当前点
0 / 1 / 2 修改可见性
J / K 切换关键点
A / D 切换图片
L / N / R / P 修改帧状态
        </div>
      </div>
    </aside>
  </div>

  <script>
    const KEYPOINT_NAMES = __KEYPOINTS__;
    const SKELETON_EDGES = __SKELETON__;

    const state = {
      items: [],
      currentIndex: 0,
      annotation: null,
      image: new Image(),
      scale: 1,
      undoStack: [],
      selectedKeypoint: 0,
    };

    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const metaEl = document.getElementById("meta");
    const statusEl = document.getElementById("statusText");
    const fileCounterEl = document.getElementById("fileCounter");
    const keypointListEl = document.getElementById("keypointList");

    function currentPath() {
      return state.items[state.currentIndex]?.path || "";
    }

    function currentStatus() {
      return state.annotation?.metadata?.frame_status || "pending";
    }

    function pushUndo() {
      if (!state.annotation) return;
      state.undoStack.push(JSON.parse(JSON.stringify(state.annotation)));
      if (state.undoStack.length > 50) {
        state.undoStack = state.undoStack.slice(-50);
      }
    }

    function setInfo() {
      const item = state.items[state.currentIndex];
      if (!item || !state.annotation) return;
      fileCounterEl.textContent = `${state.currentIndex + 1} / ${state.items.length}`;
      metaEl.textContent =
        `${item.path}\nclip=${state.annotation.clip_id || ""}, frame=${state.annotation.frame_index || 0}\nimage=${state.annotation.image_path || ""}`;
      statusEl.textContent = `当前状态: ${currentStatus()}`;
    }

    function pointColor(visibility, selected) {
      if (selected) return "#ff8000";
      if (visibility === 2) return "#22c55e";
      if (visibility === 1) return "#facc15";
      return "#ef4444";
    }

    function renderKeypointList() {
      if (!state.annotation) return;
      keypointListEl.innerHTML = "";
      KEYPOINT_NAMES.forEach((name, index) => {
        const point = state.annotation.points[name];
        const row = document.createElement("div");
        row.className = `kp ${index === state.selectedKeypoint ? "active" : ""}`;
        row.onclick = () => {
          state.selectedKeypoint = index;
          renderKeypointList();
          renderCanvas();
        };
        const marker = document.createElement("strong");
        marker.textContent = index === state.selectedKeypoint ? ">" : String(index + 1);
        const label = document.createElement("div");
        label.innerHTML = `<div>${name}</div><small>v=${point.visibility}</small>`;
        const coord = document.createElement("small");
        coord.textContent = point.x == null || point.y == null ? "--" : `(${Math.round(point.x)}, ${Math.round(point.y)})`;
        row.append(marker, label, coord);
        keypointListEl.appendChild(row);
      });
    }

    function renderCanvas() {
      if (!state.annotation || !state.image.width) return;
      const maxWidth = Math.min(window.innerWidth - 420, 1200);
      const maxHeight = Math.min(window.innerHeight - 120, 880);
      state.scale = Math.min(maxWidth / state.image.width, maxHeight / state.image.height, 1);
      canvas.width = Math.max(1, Math.round(state.image.width * state.scale));
      canvas.height = Math.max(1, Math.round(state.image.height * state.scale));
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(state.image, 0, 0, canvas.width, canvas.height);

      const scaled = {};
      for (const name of KEYPOINT_NAMES) {
        const point = state.annotation.points[name];
        if (point.x == null || point.y == null) continue;
        scaled[name] = [point.x * state.scale, point.y * state.scale];
      }

      ctx.lineWidth = 2;
      ctx.strokeStyle = "#2dd4bf";
      for (const [start, end] of SKELETON_EDGES) {
        if (!scaled[start] || !scaled[end]) continue;
        ctx.beginPath();
        ctx.moveTo(...scaled[start]);
        ctx.lineTo(...scaled[end]);
        ctx.stroke();
      }

      KEYPOINT_NAMES.forEach((name, index) => {
        const point = state.annotation.points[name];
        if (point.x == null || point.y == null) return;
        const [x, y] = scaled[name];
        ctx.fillStyle = pointColor(point.visibility, index === state.selectedKeypoint);
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        const radius = index === state.selectedKeypoint ? 6 : 5;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = "#ffffff";
        ctx.font = "12px sans-serif";
        ctx.fillText(String(index + 1), x + 8, y - 8);
      });
    }

    function updateAndRender() {
      setInfo();
      renderKeypointList();
      renderCanvas();
    }

    async function fetchJson(url, options = {}) {
      const response = await fetch(url, options);
      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || `HTTP ${response.status}`);
      }
      return response.json();
    }

    async function loadList() {
      const data = await fetchJson("/api/list");
      state.items = data.items;
      if (!state.items.length) {
        throw new Error("没有找到标注文件。");
      }
      await loadAnnotation(0);
    }

    async function loadAnnotation(index) {
      state.currentIndex = Math.max(0, Math.min(index, state.items.length - 1));
      const item = state.items[state.currentIndex];
      state.undoStack = [];
      state.annotation = await fetchJson(`/api/item?path=${encodeURIComponent(item.path)}`);
      await new Promise((resolve, reject) => {
        state.image.onload = resolve;
        state.image.onerror = () => reject(new Error("图片加载失败"));
        state.image.src = `/api/frame?path=${encodeURIComponent(state.annotation.image_path)}`;
      });
      updateAndRender();
    }

    async function saveAnnotation() {
      if (!state.annotation) return;
      await fetchJson(`/api/item?path=${encodeURIComponent(currentPath())}`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(state.annotation),
      });
      state.items[state.currentIndex].status = currentStatus();
      setInfo();
    }

    function setFrameStatus(status) {
      if (!state.annotation) return;
      pushUndo();
      state.annotation.metadata = state.annotation.metadata || {};
      state.annotation.metadata.frame_status = status;
      updateAndRender();
    }

    function markNoSwimmer() {
      if (!state.annotation) return;
      pushUndo();
      for (const name of KEYPOINT_NAMES) {
        state.annotation.points[name] = {x: null, y: null, visibility: 0};
      }
      state.annotation.metadata.frame_status = "no_swimmer";
      updateAndRender();
    }

    function clearSelectedPoint() {
      if (!state.annotation) return;
      pushUndo();
      const name = KEYPOINT_NAMES[state.selectedKeypoint];
      state.annotation.points[name] = {x: null, y: null, visibility: 0};
      updateAndRender();
    }

    function clearFrame() {
      if (!state.annotation) return;
      pushUndo();
      for (const name of KEYPOINT_NAMES) {
        state.annotation.points[name] = {x: null, y: null, visibility: 0};
      }
      state.annotation.metadata.frame_status = "pending";
      updateAndRender();
    }

    function setVisibility(value) {
      if (!state.annotation) return;
      pushUndo();
      const name = KEYPOINT_NAMES[state.selectedKeypoint];
      const point = state.annotation.points[name];
      point.visibility = value;
      if (value === 0) {
        point.x = null;
        point.y = null;
      } else if (currentStatus() === "no_swimmer") {
        state.annotation.metadata.frame_status = "labeled";
      }
      updateAndRender();
    }

    function undo() {
      const previous = state.undoStack.pop();
      if (!previous) return;
      state.annotation = previous;
      updateAndRender();
    }

    canvas.addEventListener("click", (event) => {
      if (!state.annotation) return;
      const rect = canvas.getBoundingClientRect();
      const x = (event.clientX - rect.left) / state.scale;
      const y = (event.clientY - rect.top) / state.scale;
      pushUndo();
      const name = KEYPOINT_NAMES[state.selectedKeypoint];
      const point = state.annotation.points[name];
      point.x = Math.round(x * 10) / 10;
      point.y = Math.round(y * 10) / 10;
      if (point.visibility === 0) {
        point.visibility = 2;
      }
      if (["pending", "no_swimmer"].includes(currentStatus())) {
        state.annotation.metadata.frame_status = "labeled";
      }
      updateAndRender();
    });

    canvas.addEventListener("contextmenu", (event) => {
      event.preventDefault();
      clearSelectedPoint();
    });

    document.getElementById("prevBtn").onclick = async () => { await saveAnnotation(); await loadAnnotation(state.currentIndex - 1); };
    document.getElementById("nextBtn").onclick = async () => { await saveAnnotation(); await loadAnnotation(state.currentIndex + 1); };
    document.getElementById("saveBtn").onclick = saveAnnotation;
    document.getElementById("undoBtn").onclick = undo;
    document.getElementById("clearBtn").onclick = clearSelectedPoint;
    document.getElementById("clearFrameBtn").onclick = clearFrame;
    document.getElementById("labeledBtn").onclick = () => setFrameStatus("labeled");
    document.getElementById("noSwimmerBtn").onclick = markNoSwimmer;
    document.getElementById("reviewBtn").onclick = () => setFrameStatus("review");
    document.getElementById("pendingBtn").onclick = () => setFrameStatus("pending");

    window.addEventListener("resize", renderCanvas);
    window.addEventListener("keydown", async (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s") {
        event.preventDefault();
        await saveAnnotation();
        return;
      }
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "z") {
        event.preventDefault();
        undo();
        return;
      }
      switch (event.key.toLowerCase()) {
        case "delete":
        case "backspace":
        case "c":
          event.preventDefault();
          clearSelectedPoint();
          break;
        case "0":
          setVisibility(0);
          break;
        case "1":
          setVisibility(1);
          break;
        case "2":
          setVisibility(2);
          break;
        case "j":
          state.selectedKeypoint = (state.selectedKeypoint + 1) % KEYPOINT_NAMES.length;
          updateAndRender();
          break;
        case "k":
          state.selectedKeypoint = (state.selectedKeypoint - 1 + KEYPOINT_NAMES.length) % KEYPOINT_NAMES.length;
          updateAndRender();
          break;
        case "a":
          await saveAnnotation();
          await loadAnnotation(state.currentIndex - 1);
          break;
        case "d":
          await saveAnnotation();
          await loadAnnotation(state.currentIndex + 1);
          break;
        case "l":
          setFrameStatus("labeled");
          break;
        case "n":
          markNoSwimmer();
          break;
        case "r":
          setFrameStatus("review");
          break;
        case "p":
          setFrameStatus("pending");
          break;
      }
    });

    loadList().catch((error) => {
      metaEl.textContent = `加载失败: ${error.message}`;
      console.error(error);
    });
  </script>
</body>
</html>
"""


class AnnotationWebApp:
    def __init__(self, annotation_root: str | Path, frame_root: str | Path) -> None:
        self.annotation_root = Path(annotation_root).resolve()
        self.frame_root = Path(frame_root).resolve()
        self.annotation_paths = sorted(self.annotation_root.rglob("*.json"))
        if not self.annotation_paths:
            raise ValueError(f"No annotation JSON files found under {self.annotation_root}")

    def list_items(self) -> list[dict[str, object]]:
        items: list[dict[str, object]] = []
        for path in self.annotation_paths:
            data = read_json(path)
            items.append(
                {
                    "path": str(path.relative_to(self.annotation_root).as_posix()),
                    "clip_id": data.get("clip_id", ""),
                    "frame_index": data.get("frame_index", 0),
                    "status": data.get("metadata", {}).get("frame_status", "pending"),
                    "image_path": data.get("image_path", ""),
                }
            )
        return items

    def load_annotation(self, relative_path: str) -> dict:
        return read_json(self._resolve_annotation_path(relative_path))

    def save_annotation(self, relative_path: str, data: dict) -> None:
        errors = validate_annotation(data)
        if errors:
            raise ValueError("; ".join(errors))
        write_json(self._resolve_annotation_path(relative_path), data)

    def resolve_frame_path(self, image_path: str) -> Path:
        path = Path(image_path)
        if path.is_absolute():
            if not path.exists():
                raise FileNotFoundError(image_path)
            return path
        candidate = (self.frame_root / path).resolve()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(image_path)

    def _resolve_annotation_path(self, relative_path: str) -> Path:
        candidate = (self.annotation_root / relative_path).resolve()
        if not str(candidate).startswith(str(self.annotation_root)):
            raise ValueError("Annotation path escapes annotation root.")
        if not candidate.exists():
            raise FileNotFoundError(relative_path)
        return candidate


def run_annotation_web(
    annotation_root: str | Path,
    frame_root: str | Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    app = AnnotationWebApp(annotation_root, frame_root)
    handler = _build_handler(app)
    server = ThreadingHTTPServer((host, port), handler)
    actual_host, actual_port = server.server_address
    url = f"http://{actual_host}:{actual_port}"
    print(f"Annotation web UI running at {url}")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _build_handler(app: AnnotationWebApp) -> type[BaseHTTPRequestHandler]:
    page = (
        HTML_PAGE.replace("__KEYPOINTS__", json.dumps(KEYPOINT_NAMES, ensure_ascii=False))
        .replace("__SKELETON__", json.dumps(SKELETON_EDGES, ensure_ascii=False))
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(page)
                return
            if parsed.path == "/api/list":
                self._send_json({"items": app.list_items(), "statuses": FRAME_STATUSES})
                return
            if parsed.path == "/api/item":
                relative_path = self._query_value(parsed.query, "path")
                self._send_json(app.load_annotation(relative_path))
                return
            if parsed.path == "/api/frame":
                image_path = self._query_value(parsed.query, "path")
                self._send_file(app.resolve_frame_path(image_path))
                return
            if parsed.path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/api/item":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            relative_path = self._query_value(parsed.query, "path")
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = self.rfile.read(content_length)
            data = json.loads(payload.decode("utf-8"))
            app.save_annotation(relative_path, data)
            self._send_json({"ok": True})

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

        def _query_value(self, query: str, key: str) -> str:
            values = parse_qs(query).get(key, [])
            if not values:
                raise ValueError(f"Missing required query parameter: {key}")
            return values[0]

        def _send_html(self, body: str) -> None:
            payload = body.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, data: dict) -> None:
            payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_file(self, path: Path) -> None:
            mime_type, _ = mimetypes.guess_type(str(path))
            payload = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime_type or "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
            if message is None:
                message = HTTPStatus(code).phrase
            payload = json.dumps({"error": message}, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return Handler
