## 1. CLI And Backend Scaffolding

- [x] 1.1 Add a `swim-pose predictions web` CLI entrypoint with arguments for `--predictions`, `--frame-root`, optional `--report`, and local server controls.
- [x] 1.2 Implement prediction-viewer data loading that parses prediction JSONL, validates required inputs, indexes frames by clip and order, and loads optional report JSON.
- [x] 1.3 Implement a read-only local HTTP server module for the prediction viewer, including endpoints for frame lists, selected prediction payloads, report data, and frame image bytes.

## 2. Browser Viewer Experience

- [x] 2.1 Build the browser page layout for single-model inspection with frame navigation, clip filtering, and a main canvas overlay for predicted skeleton rendering.
- [x] 2.2 Add current-frame metadata and a full 18-keypoint detail panel showing coordinates, confidence values, and visibility predictions.
- [x] 2.3 Add confidence-threshold controls and visual treatment so low-confidence or invisible points can be de-emphasized without mutating the underlying data.
- [x] 2.4 Render optional overall and per-joint metrics from the evaluation report, plus a clear empty state when no report is provided.

## 3. Documentation And Validation

- [x] 3.1 Update the README with the single-model viewer workflow and example launch commands after `predict` and `evaluate`.
- [x] 3.2 Validate the viewer manually against existing artifacts in `artifacts/predictions/`, `artifacts/reports/`, and `data/frames/`, including missing-asset and no-report behavior.
