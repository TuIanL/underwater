from __future__ import annotations

import json
import mimetypes
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .constants import KEYPOINT_SPECS, KEYPOINT_NAMES, SKELETON_EDGES, VISIBILITY_STATES
from .io import read_json, read_jsonl


HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Swim Pose Prediction Viewer</title>
  <style>
    :root {
      --bg: #09111d;
      --bg-accent: #111f34;
      --panel: rgba(8, 15, 27, 0.86);
      --panel-2: rgba(16, 27, 43, 0.92);
      --line: rgba(148, 163, 184, 0.18);
      --text: #e2e8f0;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --accent-2: #f59e0b;
      --good: #22c55e;
      --danger: #f87171;
      --shadow: 0 24px 70px rgba(2, 6, 23, 0.45);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "PingFang SC", "Noto Sans SC", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(56, 189, 248, 0.12), transparent 28%),
        radial-gradient(circle at top right, rgba(245, 158, 11, 0.10), transparent 30%),
        linear-gradient(180deg, var(--bg), var(--bg-accent));
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(720px, 1fr) 420px;
      gap: 18px;
      padding: 18px;
      min-height: 100vh;
    }
    .panel {
      border: 1px solid var(--line);
      border-radius: 22px;
      overflow: hidden;
      background: var(--panel);
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
    }
    .viewer {
      display: grid;
      grid-template-rows: auto auto 1fr;
    }
    .viewer-header,
    .sidebar-header {
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(15, 23, 42, 0.88), rgba(15, 23, 42, 0.52));
    }
    .topline {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
    }
    .viewer-header h1,
    .sidebar-header h2,
    .section h3 {
      margin: 0;
    }
    .viewer-header h1,
    .sidebar-header h2 {
      font-size: 20px;
    }
    .viewer-meta,
    .sidebar-meta,
    .helper,
    .empty-state {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
      white-space: pre-wrap;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(56, 189, 248, 0.14);
      color: #7dd3fc;
      font-size: 12px;
      font-weight: 700;
    }
    .alert {
      margin: 12px 18px 0;
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px solid rgba(248, 113, 113, 0.24);
      background: rgba(127, 29, 29, 0.28);
      color: #fecaca;
      font-size: 13px;
      line-height: 1.5;
    }
    .alert[hidden] { display: none; }
    .canvas-wrap {
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 16px;
      min-height: calc(100vh - 172px);
    }
    canvas {
      width: 100%;
      max-height: calc(100vh - 210px);
      background:
        linear-gradient(135deg, rgba(15, 23, 42, 0.85), rgba(10, 20, 34, 0.88)),
        repeating-linear-gradient(
          45deg,
          rgba(56, 189, 248, 0.05),
          rgba(56, 189, 248, 0.05) 14px,
          rgba(15, 23, 42, 0.04) 14px,
          rgba(15, 23, 42, 0.04) 28px
        );
      border-radius: 18px;
      border: 1px solid rgba(148, 163, 184, 0.16);
    }
    .sidebar {
      display: grid;
      grid-template-rows: auto auto auto minmax(220px, 1fr) minmax(220px, 1fr);
      min-height: 0;
    }
    .section {
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      min-height: 0;
    }
    .section h3 {
      font-size: 16px;
      margin-bottom: 10px;
    }
    .controls-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .controls-grid .field,
    .controls-grid .field-wide {
      display: grid;
      gap: 6px;
      font-size: 12px;
      color: var(--muted);
    }
    .controls-grid .field-wide {
      grid-column: 1 / -1;
    }
    select,
    input[type="range"] {
      width: 100%;
    }
    select {
      appearance: none;
      border: 1px solid rgba(148, 163, 184, 0.22);
      border-radius: 12px;
      padding: 10px 12px;
      background: var(--panel-2);
      color: var(--text);
      font: inherit;
    }
    .range-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: center;
    }
    .range-row strong {
      min-width: 70px;
      text-align: right;
    }
    input[type="range"] {
      accent-color: var(--accent);
    }
    button {
      border: 0;
      border-radius: 14px;
      padding: 11px 13px;
      background: linear-gradient(180deg, rgba(37, 99, 235, 0.95), rgba(29, 78, 216, 0.86));
      color: white;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      transition: transform 120ms ease, opacity 120ms ease;
    }
    button.secondary {
      background: rgba(30, 41, 59, 0.88);
      color: var(--text);
      border: 1px solid rgba(148, 163, 184, 0.16);
    }
    button:hover:not(:disabled) {
      transform: translateY(-1px);
    }
    button:disabled {
      cursor: not-allowed;
      opacity: 0.46;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .metric-card {
      padding: 12px;
      border-radius: 16px;
      background: linear-gradient(180deg, rgba(15, 23, 42, 0.94), rgba(15, 23, 42, 0.72));
      border: 1px solid rgba(148, 163, 184, 0.14);
    }
    .metric-card strong {
      display: block;
      font-size: 18px;
      margin-top: 6px;
      color: #f8fafc;
    }
    .metric-card small {
      color: var(--muted);
      font-size: 12px;
    }
    .metric-list {
      max-height: 220px;
      overflow: auto;
      margin-top: 12px;
      padding-right: 4px;
    }
    .scroll-list {
      overflow: auto;
      padding-right: 4px;
    }
    .joint-row,
    .frame-row,
    .point-row {
      display: grid;
      gap: 8px;
      padding: 11px 12px;
      border-radius: 14px;
      margin-bottom: 8px;
      background: rgba(15, 23, 42, 0.5);
      border: 1px solid transparent;
    }
    .frame-row,
    .point-row {
      cursor: pointer;
    }
    .frame-row.active,
    .point-row.active {
      border-color: rgba(56, 189, 248, 0.55);
      background: rgba(8, 47, 73, 0.34);
    }
    .point-row.dim {
      opacity: 0.72;
    }
    .frame-head,
    .point-head,
    .joint-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
    }
    .point-name,
    .frame-name,
    .joint-name {
      font-weight: 700;
    }
    .point-meta,
    .frame-meta,
    .joint-meta {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .chip-row {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
      background: rgba(56, 189, 248, 0.13);
      color: #93c5fd;
    }
    .chip.warn {
      background: rgba(245, 158, 11, 0.16);
      color: #fcd34d;
    }
    .chip.muted {
      background: rgba(148, 163, 184, 0.14);
      color: #cbd5e1;
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      flex: 0 0 auto;
      box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.04);
    }
    .section-footer {
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }
    body.player-mode .layout {
      grid-template-columns: minmax(0, 1fr) 340px;
    }
    body.player-mode #frameListSection {
      display: none;
    }
    body.player-mode .sidebar {
      grid-template-rows: auto auto auto minmax(220px, 1fr);
    }
    body.player-mode .viewer-header h1::after {
      content: " · 播放器模式";
      color: #7dd3fc;
      font-size: 14px;
      font-weight: 700;
      margin-left: 8px;
    }
    @media (max-width: 1220px) {
      .layout {
        grid-template-columns: 1fr;
      }
      .canvas-wrap {
        min-height: auto;
      }
      canvas {
        max-height: 68vh;
      }
      .sidebar {
        grid-template-rows: auto auto auto minmax(220px, 360px) minmax(220px, 360px);
      }
    }
  </style>
</head>
<body>
  <div class="layout">
    <section class="panel viewer">
      <div class="viewer-header">
        <div class="topline">
          <h1>模型预测查看</h1>
          <span id="frameCounter" class="badge">0 / 0</span>
        </div>
        <div id="viewerMeta" class="viewer-meta">正在加载预测结果...</div>
      </div>
      <div id="errorBanner" class="alert" hidden></div>
      <div class="canvas-wrap">
        <canvas id="viewerCanvas"></canvas>
      </div>
    </section>

    <aside class="panel sidebar">
      <div class="sidebar-header">
        <h2>单模型结果</h2>
        <div id="sidebarMeta" class="sidebar-meta">正在初始化...</div>
      </div>

      <section class="section">
        <h3>浏览控制</h3>
        <div class="controls-grid">
          <label class="field-wide">
            <span>Clip 筛选</span>
            <select id="clipFilter"></select>
          </label>

          <button id="playPauseBtn">开始播放</button>
          <button id="restartBtn" class="secondary">回到开头</button>

          <button id="prevBtn" class="secondary">上一帧</button>
          <button id="nextBtn">下一帧</button>

          <button id="loopBtn" class="secondary">循环: 开</button>
          <label class="field">
            <span>播放速度</span>
            <div class="range-row">
              <input id="playbackFpsInput" type="range" min="2" max="30" step="1" value="12">
              <strong id="playbackFpsValue">12 FPS</strong>
            </div>
          </label>

          <label class="field-wide">
            <span>播放进度</span>
            <div class="range-row">
              <input id="timelineInput" type="range" min="0" max="0" step="1" value="0">
              <strong id="timelineValue">0 / 0</strong>
            </div>
          </label>

          <label class="field-wide">
            <span>高亮阈值</span>
            <div class="range-row">
              <input id="thresholdInput" type="range" min="0" max="1" step="0.01" value="0">
              <strong id="thresholdValue">0.00</strong>
            </div>
          </label>
        </div>
        <div id="playbackHint" class="section-footer">
          低于阈值的关键点和骨架连线会被淡化，可见性为 0 的点会继续显示为弱提示。
        </div>
      </section>

      <section class="section">
        <h3>评估摘要</h3>
        <div id="overallMetrics" class="metric-grid"></div>
        <div id="perJointList" class="metric-list"></div>
        <div id="reportNote" class="section-footer"></div>
      </section>

      <section class="section">
        <h3>关键点详情</h3>
        <div id="keypointList" class="scroll-list"></div>
      </section>

      <section id="frameListSection" class="section">
        <h3>帧列表</h3>
        <div id="frameList" class="scroll-list"></div>
      </section>
    </aside>
  </div>

  <script>
    const KEYPOINT_NAMES = __KEYPOINTS__;
    const KEYPOINT_GROUPS = __KEYPOINT_GROUPS__;
    const SKELETON_EDGES = __SKELETON__;
    const VISIBILITY_STATES = __VISIBILITY_STATES__;
    const INITIAL_PLAYER_MODE = __INITIAL_PLAYER_MODE__;
    const INITIAL_CLIP = __INITIAL_CLIP__;
    const GROUP_COLORS = {
      head: "#fb923c",
      upper: "#34d399",
      lower: "#38bdf8",
      foot: "#facc15",
      unknown: "#cbd5e1",
    };

    const state = {
      items: [],
      filteredItems: [],
      currentIndex: -1,
      prediction: null,
      report: null,
      selectedClip: "__all__",
      threshold: 0,
      image: new Image(),
      frameError: "",
      sessionMeta: null,
      playbackFps: 12,
      isPlaying: false,
      playbackTimer: null,
      loopPlayback: true,
      imageCache: new Map(),
      playerMode: INITIAL_PLAYER_MODE,
      initialClip: INITIAL_CLIP || "",
    };

    const canvas = document.getElementById("viewerCanvas");
    const ctx = canvas.getContext("2d");
    const viewerMetaEl = document.getElementById("viewerMeta");
    const sidebarMetaEl = document.getElementById("sidebarMeta");
    const frameCounterEl = document.getElementById("frameCounter");
    const errorBannerEl = document.getElementById("errorBanner");
    const clipFilterEl = document.getElementById("clipFilter");
    const playPauseBtn = document.getElementById("playPauseBtn");
    const restartBtn = document.getElementById("restartBtn");
    const thresholdInputEl = document.getElementById("thresholdInput");
    const thresholdValueEl = document.getElementById("thresholdValue");
    const playbackFpsInputEl = document.getElementById("playbackFpsInput");
    const playbackFpsValueEl = document.getElementById("playbackFpsValue");
    const timelineInputEl = document.getElementById("timelineInput");
    const timelineValueEl = document.getElementById("timelineValue");
    const playbackHintEl = document.getElementById("playbackHint");
    const loopBtn = document.getElementById("loopBtn");
    const overallMetricsEl = document.getElementById("overallMetrics");
    const perJointListEl = document.getElementById("perJointList");
    const reportNoteEl = document.getElementById("reportNote");
    const keypointListEl = document.getElementById("keypointList");
    const frameListEl = document.getElementById("frameList");
    const frameListSectionEl = document.getElementById("frameListSection");
    const prevBtn = document.getElementById("prevBtn");
    const nextBtn = document.getElementById("nextBtn");
    const IMAGE_CACHE_LIMIT = 18;

    function currentItem() {
      return state.currentIndex >= 0 ? state.items[state.currentIndex] || null : null;
    }

    function filteredPosition() {
      return state.filteredItems.findIndex((item) => item.index === state.currentIndex);
    }

    function setError(message) {
      errorBannerEl.textContent = message || "";
      errorBannerEl.hidden = !message;
    }

    async function fetchJson(url, options = undefined) {
      const response = await fetch(url, options);
      const contentType = response.headers.get("content-type") || "";
      const payload = contentType.includes("application/json") ? await response.json() : null;
      if (!response.ok) {
        throw new Error(payload?.error || `Request failed (${response.status})`);
      }
      return payload;
    }

    function formatNumber(value, digits = 3) {
      if (typeof value !== "number" || Number.isNaN(value)) return "--";
      return value.toFixed(digits);
    }

    function formatCoordinate(point) {
      if (!point || point.x == null || point.y == null) return "(--, --)";
      return `(${formatNumber(point.x, 1)}, ${formatNumber(point.y, 1)})`;
    }

    function pointGroup(name) {
      return KEYPOINT_GROUPS[name] || "unknown";
    }

    function pointColor(name) {
      return GROUP_COLORS[pointGroup(name)] || GROUP_COLORS.unknown;
    }

    function visibilityLabel(value) {
      return VISIBILITY_STATES[String(value)] || "unknown";
    }

    function pointAlpha(point) {
      if (!point) return 0;
      const confidence = typeof point.confidence === "number" ? point.confidence : 0;
      if ((point.visibility ?? 0) === 0) {
        return Math.max(0.12, confidence * 0.35);
      }
      if (confidence >= state.threshold) {
        return Math.max(0.55, confidence);
      }
      return Math.max(0.15, confidence * 0.35);
    }

    function renderCounter() {
      const position = filteredPosition();
      const total = state.filteredItems.length;
      const current = position >= 0 ? position + 1 : 0;
      frameCounterEl.textContent = `${current} / ${total}`;
    }

    function renderSidebarMeta() {
      if (!state.sessionMeta) {
        sidebarMetaEl.textContent = "正在加载会话信息...";
        return;
      }
      const reportText = state.sessionMeta.report_available
        ? `评估报告: ${state.sessionMeta.report_file}`
        : "评估报告: 未提供";
      const modeText = state.playerMode
        ? `模式: 纯播放器 (${state.selectedClip === "__all__" ? "自动 clip" : state.selectedClip})`
        : "模式: 检查器";
      sidebarMetaEl.textContent =
        `预测文件: ${state.sessionMeta.prediction_file}\n` +
        `总帧数: ${state.items.length}\n` +
        `${modeText}\n` +
        reportText;
    }

    function renderViewerMeta() {
      const item = currentItem();
      if (!item || !state.prediction) {
        viewerMetaEl.textContent = "请选择一帧查看模型输出。";
        return;
      }
      const lines = [
        `clip=${item.clip_id || "unknown"} | frame=${item.frame_index} | view=${item.source_view || "unknown"}`,
        `athlete=${item.athlete_id || "unknown"} | session=${item.session_id || "unknown"}`,
        `image=${item.image_path}`,
        `playback=${state.isPlaying ? "playing" : "paused"} | fps=${state.playbackFps}`,
      ];
      if (state.frameError) {
        lines.push(`frame_error=${state.frameError}`);
      }
      viewerMetaEl.textContent = lines.join("\\n");
    }

    function updateNavigationButtons() {
      const position = filteredPosition();
      prevBtn.disabled = position <= 0;
      nextBtn.disabled = position < 0 || position >= state.filteredItems.length - 1;
      restartBtn.disabled = !state.filteredItems.length;
    }

    function renderClipFilter() {
      const options = ['<option value="__all__">全部 Clip</option>']
        .concat(state.sessionMeta?.clips?.map((clip) => `<option value="${clip}">${clip}</option>`) || []);
      clipFilterEl.innerHTML = options.join("");
      clipFilterEl.value = state.selectedClip;
    }

    function renderOverallMetrics() {
      const report = state.report?.report;
      if (!report?.overall) {
        overallMetricsEl.innerHTML = '<div class="empty-state">当前会话没有附带评估报告，仍然可以查看逐帧预测结果。</div>';
        perJointListEl.innerHTML = "";
        reportNoteEl.textContent = "如果你提供 evaluate 生成的 JSON，这里会显示 overall 和 per-joint 指标。";
        return;
      }
      const overall = report.overall;
      const cards = Object.entries(overall).map(([key, value]) => {
        return `
          <div class="metric-card">
            <small>${key}</small>
            <strong>${formatNumber(value, 4)}</strong>
          </div>
        `;
      });
      overallMetricsEl.innerHTML = cards.join("");
      const perJoint = report.per_joint || {};
      perJointListEl.innerHTML = KEYPOINT_NAMES
        .filter((name) => perJoint[name])
        .map((name) => {
          const metrics = perJoint[name];
          return `
            <div class="joint-row">
              <div class="joint-head">
                <span class="joint-name">${name}</span>
                <span class="chip muted">count ${metrics.count ?? "--"}</span>
              </div>
              <div class="joint-meta">mean_normalized_error = ${formatNumber(metrics.mean_normalized_error, 4)}</div>
            </div>
          `;
        })
        .join("");
      reportNoteEl.textContent = state.sessionMeta?.report_file
        ? `已加载 ${state.sessionMeta.report_file}`
        : "已加载评估报告";
    }

    function estimateFrameStep(items) {
      const steps = [];
      for (let i = 1; i < items.length; i += 1) {
        const previous = items[i - 1];
        const current = items[i];
        if (previous.clip_id !== current.clip_id || previous.source_view !== current.source_view) {
          continue;
        }
        const step = Number(current.frame_index) - Number(previous.frame_index);
        if (step > 0) {
          steps.push(step);
        }
      }
      if (!steps.length) {
        return 1;
      }
      steps.sort((left, right) => left - right);
      return steps[Math.floor(steps.length / 2)] || 1;
    }

    function renderPlaybackControls() {
      playPauseBtn.textContent = state.isPlaying ? "暂停播放" : "开始播放";
      loopBtn.textContent = `循环: ${state.loopPlayback ? "开" : "关"}`;
      playPauseBtn.disabled = state.filteredItems.length <= 1;
      loopBtn.disabled = !state.filteredItems.length;
      playbackFpsValueEl.textContent = `${state.playbackFps} FPS`;
    }

    function renderPlaybackHint() {
      const step = estimateFrameStep(state.filteredItems);
      const cadenceNote = step > 1
        ? `当前预测序列大约每 ${step} 帧采样一次，所以这是抽帧回放，不是原视频逐帧重建。`
        : "当前序列接近逐帧预测，更适合连续播放观看。";
      const playNote = state.isPlaying
        ? `正在以 ${state.playbackFps} FPS 连续播放。`
        : `准备以 ${state.playbackFps} FPS 播放。`;
      const modeNote = state.playerMode
        ? "播放器模式会按当前 clip 自动连续播放，并隐藏右侧长帧列表。"
        : "检查器模式保留完整帧列表，适合逐帧排查。";
      playbackHintEl.textContent =
        `${modeNote} ${cadenceNote} ${playNote} 低于阈值的关键点和骨架连线会被淡化，可见性为 0 的点会继续显示为弱提示。`;
    }

    function applyModeUi() {
      document.body.classList.toggle("player-mode", state.playerMode);
      frameListSectionEl.hidden = state.playerMode;
    }

    function updateTimelineControl() {
      const total = state.filteredItems.length;
      const position = filteredPosition();
      timelineInputEl.max = String(Math.max(total - 1, 0));
      timelineInputEl.disabled = !total;
      timelineInputEl.value = String(Math.max(position, 0));
      timelineValueEl.textContent = total ? `${Math.max(position + 1, 1)} / ${total}` : "0 / 0";
    }

    function syncActiveFrameRow() {
      const previous = frameListEl.querySelector(".frame-row.active");
      if (previous) {
        previous.classList.remove("active");
      }
      const current = frameListEl.querySelector(`[data-index="${state.currentIndex}"]`);
      if (current) {
        current.classList.add("active");
      }
    }

    function renderFrameList() {
      if (!state.filteredItems.length) {
        frameListEl.innerHTML = '<div class="empty-state">当前筛选条件下没有可浏览的帧。</div>';
        return;
      }
      frameListEl.innerHTML = state.filteredItems.map((item) => {
        const isActive = item.index === state.currentIndex;
        return `
          <div class="frame-row ${isActive ? "active" : ""}" data-index="${item.index}">
            <div class="frame-head">
              <span class="frame-name">Frame ${String(item.frame_index).padStart(6, "0")}</span>
              <span class="chip muted">${item.source_view || "unknown"}</span>
            </div>
            <div class="frame-meta">${item.clip_id || "unknown"}\\n${item.image_path}</div>
          </div>
        `;
      }).join("");
      frameListEl.querySelectorAll("[data-index]").forEach((node) => {
        node.addEventListener("click", () => selectFrame(Number(node.dataset.index)));
      });
      syncActiveFrameRow();
    }

    function renderKeypointList() {
      if (!state.prediction?.points) {
        keypointListEl.innerHTML = '<div class="empty-state">等待载入关键点结果。</div>';
        return;
      }
      keypointListEl.innerHTML = KEYPOINT_NAMES.map((name) => {
        const point = state.prediction.points[name];
        const lowConfidence = typeof point.confidence === "number" && point.confidence < state.threshold;
        const alpha = pointAlpha(point);
        const visibility = visibilityLabel(point.visibility);
        const chips = [
          `<span class="chip">${visibility}</span>`,
          `<span class="chip muted">conf ${formatNumber(point.confidence, 3)}</span>`,
        ];
        if (lowConfidence) {
          chips.push('<span class="chip warn">低于阈值</span>');
        }
        if ((point.visibility ?? 0) === 0) {
          chips.push('<span class="chip muted">弱提示</span>');
        }
        return `
          <div class="point-row ${lowConfidence || (point.visibility ?? 0) === 0 ? "dim" : ""}">
            <div class="point-head">
              <span class="point-name">
                <span class="dot" style="background:${pointColor(name)}; opacity:${alpha};"></span>
                ${name}
              </span>
              <span class="chip muted">${pointGroup(name)}</span>
            </div>
            <div class="point-meta">${formatCoordinate(point)}</div>
            <div class="chip-row">${chips.join("")}</div>
          </div>
        `;
      }).join("");
    }

    function drawEmptyCanvas(message) {
      canvas.width = 1280;
      canvas.height = 720;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
      gradient.addColorStop(0, "#0f172a");
      gradient.addColorStop(1, "#0b1a2e");
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = "rgba(148, 163, 184, 0.18)";
      ctx.lineWidth = 2;
      ctx.strokeRect(30, 30, canvas.width - 60, canvas.height - 60);
      ctx.fillStyle = "#e2e8f0";
      ctx.font = '700 30px "PingFang SC", "Noto Sans SC", sans-serif';
      ctx.textAlign = "center";
      ctx.fillText(message, canvas.width / 2, canvas.height / 2 - 12);
      ctx.fillStyle = "#94a3b8";
      ctx.font = '400 18px "PingFang SC", "Noto Sans SC", sans-serif';
      ctx.fillText("加载成功后会在这里显示原图和预测骨架。", canvas.width / 2, canvas.height / 2 + 28);
    }

    function drawCanvas() {
      if (!state.prediction) {
        drawEmptyCanvas("暂无预测结果");
        return;
      }
      if (state.frameError || !state.image.complete || !state.image.naturalWidth) {
        drawEmptyCanvas(state.frameError || "正在加载帧图像");
        return;
      }

      canvas.width = state.image.naturalWidth;
      canvas.height = state.image.naturalHeight;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(state.image, 0, 0, canvas.width, canvas.height);

      for (const [startName, endName] of SKELETON_EDGES) {
        const start = state.prediction.points[startName];
        const end = state.prediction.points[endName];
        if (!start || !end || start.x == null || start.y == null || end.x == null || end.y == null) {
          continue;
        }
        const alpha = Math.min(pointAlpha(start), pointAlpha(end));
        ctx.save();
        ctx.strokeStyle = "rgba(56, 189, 248, 0.86)";
        ctx.globalAlpha = alpha;
        ctx.lineWidth = alpha > 0.45 ? 5 : 3;
        if ((start.visibility ?? 0) === 0 || (end.visibility ?? 0) === 0) {
          ctx.setLineDash([10, 10]);
        }
        ctx.beginPath();
        ctx.moveTo(start.x, start.y);
        ctx.lineTo(end.x, end.y);
        ctx.stroke();
        ctx.restore();
      }

      for (const name of KEYPOINT_NAMES) {
        const point = state.prediction.points[name];
        if (!point || point.x == null || point.y == null) {
          continue;
        }
        const alpha = pointAlpha(point);
        const color = pointColor(name);
        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.fillStyle = color;
        ctx.strokeStyle = "#f8fafc";
        ctx.lineWidth = 2.5;
        const radius = (point.visibility ?? 0) === 0 ? 7 : 8.5;
        ctx.beginPath();
        ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
        if ((point.visibility ?? 0) === 0) {
          ctx.stroke();
        } else {
          ctx.fill();
          ctx.stroke();
        }
        ctx.restore();
      }
    }

    function renderAll() {
      applyModeUi();
      renderSidebarMeta();
      renderViewerMeta();
      renderCounter();
      updateNavigationButtons();
      renderPlaybackControls();
      renderPlaybackHint();
      updateTimelineControl();
      renderKeypointList();
      syncActiveFrameRow();
      drawCanvas();
    }

    function applyFilter(clipId) {
      stopPlayback();
      state.selectedClip = clipId;
      state.filteredItems = clipId === "__all__"
        ? [...state.items]
        : state.items.filter((item) => item.clip_id === clipId);
      renderPlaybackHint();
      renderPlaybackControls();
      updateTimelineControl();
      renderCounter();
      updateNavigationButtons();
      renderFrameList();
    }

    function trimImageCache(preserve = []) {
      const preserveSet = new Set(preserve);
      while (state.imageCache.size > IMAGE_CACHE_LIMIT) {
        const oldestKey = state.imageCache.keys().next().value;
        if (preserveSet.has(oldestKey)) {
          const value = state.imageCache.get(oldestKey);
          state.imageCache.delete(oldestKey);
          state.imageCache.set(oldestKey, value);
          continue;
        }
        state.imageCache.delete(oldestKey);
      }
    }

    function loadImageForIndex(index) {
      if (state.imageCache.has(index)) {
        return state.imageCache.get(index);
      }
      const promise = new Promise((resolve) => {
        const nextImage = new Image();
        nextImage.onload = () => resolve({ image: nextImage, error: "" });
        nextImage.onerror = () => {
          const item = state.items[index];
          resolve({
            image: new Image(),
            error: `无法加载当前帧图像: ${item?.image_path || ""}`,
          });
        };
        nextImage.src = `/api/frame?index=${index}`;
      });
      state.imageCache.set(index, promise);
      trimImageCache([state.currentIndex, index]);
      return promise;
    }

    function primePlaybackCache(index) {
      const position = state.filteredItems.findIndex((item) => item.index === index);
      if (position < 0) {
        return;
      }
      const preserve = [index];
      for (let offset = 1; offset <= 3; offset += 1) {
        const nextItem = state.filteredItems[position + offset];
        if (!nextItem) {
          continue;
        }
        preserve.push(nextItem.index);
        void loadImageForIndex(nextItem.index);
      }
      trimImageCache(preserve);
    }

    async function preloadFrame(index) {
      state.frameError = "";
      const result = await loadImageForIndex(index);
      if (result.error) {
        state.frameError = result.error;
        setError(state.frameError);
        state.image = new Image();
        return false;
      }
      state.image = result.image;
      setError("");
      primePlaybackCache(index);
      return true;
    }

    function stopPlayback() {
      if (state.playbackTimer) {
        window.clearTimeout(state.playbackTimer);
        state.playbackTimer = null;
      }
      state.isPlaying = false;
      renderPlaybackControls();
      renderPlaybackHint();
      renderViewerMeta();
    }

    function schedulePlaybackTick() {
      if (!state.isPlaying) {
        return;
      }
      if (state.playbackTimer) {
        window.clearTimeout(state.playbackTimer);
      }
      state.playbackTimer = window.setTimeout(async () => {
        const position = filteredPosition();
        if (position < 0) {
          stopPlayback();
          return;
        }
        let nextPosition = position + 1;
        if (nextPosition >= state.filteredItems.length) {
          if (!state.loopPlayback) {
            stopPlayback();
            return;
          }
          nextPosition = 0;
        }
        await selectFrame(state.filteredItems[nextPosition].index, { preservePlayback: true });
      }, Math.max(16, 1000 / state.playbackFps));
    }

    function startPlayback() {
      if (state.filteredItems.length <= 1) {
        return;
      }
      state.isPlaying = true;
      renderPlaybackControls();
      renderPlaybackHint();
      renderViewerMeta();
      schedulePlaybackTick();
    }

    async function selectFrame(index, options = {}) {
      const preservePlayback = options.preservePlayback === true;
      try {
        if (!preservePlayback) {
          stopPlayback();
        }
        setError("");
        state.currentIndex = index;
        state.prediction = await fetchJson(`/api/item?index=${index}`);
        const imageLoaded = await preloadFrame(index);
        renderAll();
        if (preservePlayback && imageLoaded) {
          schedulePlaybackTick();
        }
        if (preservePlayback && !imageLoaded) {
          stopPlayback();
        }
      } catch (error) {
        stopPlayback();
        state.prediction = null;
        state.frameError = "";
        setError(error.message);
        renderAll();
      }
    }

    async function loadReport() {
      state.report = await fetchJson("/api/report");
      renderOverallMetrics();
    }

    async function loadSession() {
      state.sessionMeta = await fetchJson("/api/list");
      state.items = state.sessionMeta.items || [];
      state.playerMode = state.sessionMeta.player_mode === true;
      if (state.sessionMeta.initial_clip) {
        state.initialClip = state.sessionMeta.initial_clip;
      }
      if (!state.items.length) {
        setError("预测文件中没有可显示的结果。");
        renderAll();
        return;
      }
      renderClipFilter();
      const desiredClip = state.playerMode
        ? (state.initialClip || state.sessionMeta.clips?.[0] || "__all__")
        : "__all__";
      applyFilter(desiredClip);
      const firstItem = state.filteredItems[0] || state.items[0];
      if (firstItem) {
        await selectFrame(firstItem.index);
      }
      if (state.playerMode && state.filteredItems.length > 1) {
        startPlayback();
      }
    }

    prevBtn.addEventListener("click", async () => {
      const position = filteredPosition();
      if (position > 0) {
        await selectFrame(state.filteredItems[position - 1].index);
      }
    });

    nextBtn.addEventListener("click", async () => {
      const position = filteredPosition();
      if (position >= 0 && position < state.filteredItems.length - 1) {
        await selectFrame(state.filteredItems[position + 1].index);
      }
    });

    playPauseBtn.addEventListener("click", async () => {
      if (state.isPlaying) {
        stopPlayback();
        return;
      }
      if (state.currentIndex < 0 && state.filteredItems.length) {
        await selectFrame(state.filteredItems[0].index);
      }
      startPlayback();
    });

    restartBtn.addEventListener("click", async () => {
      if (!state.filteredItems.length) {
        return;
      }
      const shouldResume = state.isPlaying;
      stopPlayback();
      await selectFrame(state.filteredItems[0].index);
      if (shouldResume) {
        startPlayback();
      }
    });

    loopBtn.addEventListener("click", () => {
      state.loopPlayback = !state.loopPlayback;
      renderPlaybackControls();
      renderPlaybackHint();
    });

    clipFilterEl.addEventListener("change", async (event) => {
      applyFilter(event.target.value);
      if (!state.filteredItems.length) {
        state.currentIndex = -1;
        state.prediction = null;
        state.frameError = "";
        renderAll();
        return;
      }
      if (state.playerMode) {
        await selectFrame(state.filteredItems[0].index);
        if (state.filteredItems.length > 1) {
          startPlayback();
        }
        return;
      }
      const stillVisible = state.filteredItems.some((item) => item.index === state.currentIndex);
      await selectFrame((stillVisible ? currentItem() : state.filteredItems[0]).index);
    });

    playbackFpsInputEl.addEventListener("input", () => {
      state.playbackFps = Number(playbackFpsInputEl.value);
      renderPlaybackControls();
      renderPlaybackHint();
      renderViewerMeta();
      if (state.isPlaying) {
        schedulePlaybackTick();
      }
    });

    timelineInputEl.addEventListener("change", async () => {
      const position = Number(timelineInputEl.value);
      const item = state.filteredItems[position];
      if (item) {
        await selectFrame(item.index);
      }
    });

    thresholdInputEl.addEventListener("input", () => {
      state.threshold = Number(thresholdInputEl.value);
      thresholdValueEl.textContent = formatNumber(state.threshold, 2);
      renderKeypointList();
      drawCanvas();
    });

    window.addEventListener("keydown", async (event) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) {
        return;
      }
      if (event.key === "ArrowLeft" || event.key.toLowerCase() === "a") {
        event.preventDefault();
        prevBtn.click();
      }
      if (event.key === "ArrowRight" || event.key.toLowerCase() === "d") {
        event.preventDefault();
        nextBtn.click();
      }
      if (event.code === "Space") {
        event.preventDefault();
        playPauseBtn.click();
      }
      if (event.key.toLowerCase() === "r") {
        event.preventDefault();
        restartBtn.click();
      }
    });

    playbackFpsValueEl.textContent = `${state.playbackFps} FPS`;
    thresholdValueEl.textContent = formatNumber(state.threshold, 2);
    drawEmptyCanvas("正在初始化查看器");

    Promise.all([loadReport(), loadSession()])
      .then(() => renderAll())
      .catch((error) => {
        setError(error.message);
        renderAll();
      });
  </script>
</body>
</html>
"""


class PredictionWebApp:
    def __init__(
        self,
        predictions_path: str | Path,
        frame_root: str | Path,
        report_path: str | Path | None = None,
        initial_clip: str | None = None,
        player_mode: bool = False,
    ) -> None:
        self.predictions_path = Path(predictions_path).resolve()
        self.frame_root = Path(frame_root).resolve()
        self.report_path = Path(report_path).resolve() if report_path is not None else None
        self.initial_clip = initial_clip or ""
        self.player_mode = player_mode

        if not self.predictions_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {self.predictions_path}")
        if not self.predictions_path.is_file():
            raise ValueError(f"Predictions path is not a file: {self.predictions_path}")
        if not self.frame_root.exists():
            raise FileNotFoundError(f"Frame root not found: {self.frame_root}")
        if not self.frame_root.is_dir():
            raise ValueError(f"Frame root is not a directory: {self.frame_root}")

        self.report = self._load_report()
        self.predictions = self._load_predictions()
        self.items = self._build_items()
        self.clips = sorted({item["clip_id"] for item in self.items if item["clip_id"]})
        if self.initial_clip and self.initial_clip not in self.clips:
            raise ValueError(f"Unknown clip for player mode: {self.initial_clip}")

    def _load_predictions(self) -> list[dict]:
        rows = read_jsonl(self.predictions_path)
        if not rows:
            raise ValueError(f"No prediction rows found in {self.predictions_path}")

        normalized = [self._normalize_prediction(index, row) for index, row in enumerate(rows, start=1)]
        normalized.sort(
            key=lambda row: (
                str(row["clip_id"]),
                str(row["source_view"]),
                int(row["frame_index"]),
                str(row["image_path"]),
            )
        )
        return normalized

    def _load_report(self) -> dict | None:
        if self.report_path is None:
            return None
        if not self.report_path.exists():
            raise FileNotFoundError(f"Report file not found: {self.report_path}")
        if not self.report_path.is_file():
            raise ValueError(f"Report path is not a file: {self.report_path}")
        report = read_json(self.report_path)
        if not isinstance(report, dict):
            raise ValueError(f"Report JSON must contain an object: {self.report_path}")
        return report

    def _build_items(self) -> list[dict[str, object]]:
        return [
            {
                "index": index,
                "clip_id": prediction["clip_id"],
                "athlete_id": prediction["athlete_id"],
                "session_id": prediction["session_id"],
                "frame_index": prediction["frame_index"],
                "source_view": prediction["source_view"],
                "image_path": prediction["image_path"],
            }
            for index, prediction in enumerate(self.predictions)
        ]

    def _normalize_prediction(self, row_number: int, row: dict) -> dict:
        if not isinstance(row, dict):
            raise ValueError(f"Prediction row {row_number} must be a JSON object.")

        image_path = str(row.get("image_path", "")).strip()
        if not image_path:
            raise ValueError(f"Prediction row {row_number} is missing image_path.")

        points = row.get("points")
        if not isinstance(points, dict):
            raise ValueError(f"Prediction row {row_number} is missing points.")

        normalized_points = {}
        for keypoint_name in KEYPOINT_NAMES:
            point = points.get(keypoint_name)
            if not isinstance(point, dict):
                raise ValueError(f"Prediction row {row_number} is missing point '{keypoint_name}'.")
            visibility = self._coerce_int(point.get("visibility"), f"row {row_number} point {keypoint_name} visibility")
            if visibility not in VISIBILITY_STATES:
                raise ValueError(
                    f"Prediction row {row_number} point '{keypoint_name}' has invalid visibility: {visibility}"
                )
            normalized_points[keypoint_name] = {
                "x": self._coerce_optional_float(point.get("x")),
                "y": self._coerce_optional_float(point.get("y")),
                "confidence": self._coerce_float(point.get("confidence"), f"row {row_number} point {keypoint_name} confidence"),
                "visibility": visibility,
            }

        return {
            "annotation_path": str(row.get("annotation_path", "")),
            "clip_id": str(row.get("clip_id", "")),
            "athlete_id": str(row.get("athlete_id", "")),
            "session_id": str(row.get("session_id", "")),
            "frame_index": self._coerce_int(row.get("frame_index"), f"row {row_number} frame_index"),
            "source_view": str(row.get("source_view", "")),
            "image_path": image_path,
            "points": normalized_points,
        }

    def list_items(self) -> dict[str, object]:
        return {
            "items": self.items,
            "clips": self.clips,
            "prediction_file": self.predictions_path.name,
            "report_file": self.report_path.name if self.report_path else "",
            "report_available": self.report is not None,
            "player_mode": self.player_mode,
            "initial_clip": self.initial_clip,
        }

    def load_prediction(self, index: int) -> dict:
        if index < 0 or index >= len(self.predictions):
            raise IndexError(f"Prediction index out of range: {index}")
        return self.predictions[index]

    def report_payload(self) -> dict[str, object]:
        return {
            "available": self.report is not None,
            "report": self.report,
        }

    def resolve_frame_path(self, index: int) -> Path:
        image_path = self.load_prediction(index)["image_path"]
        path = Path(str(image_path))
        if path.is_absolute():
            if not path.exists():
                raise FileNotFoundError(f"Frame file not found: {image_path}")
            return path

        candidate = (self.frame_root / path).resolve()
        if not str(candidate).startswith(str(self.frame_root)):
            raise ValueError(f"Frame path escapes frame root: {image_path}")
        if not candidate.exists():
            raise FileNotFoundError(f"Frame file not found under frame root: {image_path}")
        return candidate

    @staticmethod
    def _coerce_int(value: object, field_name: str) -> int:
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid integer for {field_name}: {value}") from exc

    @staticmethod
    def _coerce_float(value: object, field_name: str) -> float:
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid float for {field_name}: {value}") from exc

    @staticmethod
    def _coerce_optional_float(value: object) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid float coordinate: {value}") from exc


def run_prediction_web(
    predictions_path: str | Path,
    frame_root: str | Path,
    report_path: str | Path | None = None,
    initial_clip: str | None = None,
    player_mode: bool = False,
    host: str = "127.0.0.1",
    port: int = 8766,
    open_browser: bool = True,
) -> None:
    app = PredictionWebApp(
        predictions_path=predictions_path,
        frame_root=frame_root,
        report_path=report_path,
        initial_clip=initial_clip,
        player_mode=player_mode,
    )
    handler = _build_handler(app)
    server = ThreadingHTTPServer((host, port), handler)
    actual_host, actual_port = server.server_address
    url = f"http://{actual_host}:{actual_port}"
    print(f"Prediction viewer running at {url}")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _build_handler(app: PredictionWebApp) -> type[BaseHTTPRequestHandler]:
    page = (
        HTML_PAGE.replace("__KEYPOINTS__", json.dumps(KEYPOINT_NAMES, ensure_ascii=False))
        .replace(
            "__KEYPOINT_GROUPS__",
            json.dumps({spec.name: spec.group for spec in KEYPOINT_SPECS}, ensure_ascii=False),
        )
        .replace("__SKELETON__", json.dumps(SKELETON_EDGES, ensure_ascii=False))
        .replace(
            "__VISIBILITY_STATES__",
            json.dumps({str(key): value for key, value in VISIBILITY_STATES.items()}, ensure_ascii=False),
        )
        .replace("__INITIAL_PLAYER_MODE__", json.dumps(app.player_mode))
        .replace("__INITIAL_CLIP__", json.dumps(app.initial_clip, ensure_ascii=False))
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self._send_html(page)
                    return
                if parsed.path == "/api/list":
                    self._send_json(app.list_items())
                    return
                if parsed.path == "/api/item":
                    index = self._query_index(parsed.query)
                    self._send_json(app.load_prediction(index))
                    return
                if parsed.path == "/api/report":
                    self._send_json(app.report_payload())
                    return
                if parsed.path == "/api/frame":
                    index = self._query_index(parsed.query)
                    self._send_file(app.resolve_frame_path(index))
                    return
                if parsed.path == "/favicon.ico":
                    self.send_response(HTTPStatus.NO_CONTENT)
                    self.end_headers()
                    return
                self.send_error(HTTPStatus.NOT_FOUND)
            except FileNotFoundError as exc:
                self.send_error(HTTPStatus.NOT_FOUND, str(exc))
            except IndexError as exc:
                self.send_error(HTTPStatus.NOT_FOUND, str(exc))
            except ValueError as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            except Exception as exc:  # pragma: no cover - defensive server guard
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

        def _query_index(self, query: str) -> int:
            values = parse_qs(query).get("index", [])
            if not values:
                raise ValueError("Missing required query parameter: index")
            try:
                return int(values[0])
            except ValueError as exc:
                raise ValueError(f"Invalid index value: {values[0]}") from exc

        def _send_html(self, body: str) -> None:
            payload = body.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, data: dict[str, object]) -> None:
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
