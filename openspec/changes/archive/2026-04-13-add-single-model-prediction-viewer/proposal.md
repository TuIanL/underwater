## Why

The project can already export frame-level prediction JSONL files and evaluation reports, but there is no direct way to inspect those results visually in the browser. A lightweight single-model viewer would make it much easier to verify whether the trained model is placing keypoints sensibly, spot confidence failures, and communicate model quality without manually opening raw files.

## What Changes

- Add a browser-based local viewer for one model's prediction results over swim frame images.
- Support loading a prediction JSONL file together with the corresponding frame root so users can browse frames and inspect the predicted 18-point skeleton overlay.
- Show per-frame metadata and per-keypoint confidence and visibility values for the selected frame.
- Show model-level summary metrics when an evaluation report JSON is provided alongside predictions.
- Add a CLI entrypoint for launching the viewer in the same local-web style already used by the annotation tool.

## Capabilities

### New Capabilities
- `prediction-viewer`: Defines how users open, browse, and inspect one model's prediction outputs and optional evaluation summary in a local browser UI.

### Modified Capabilities
- None.

## Impact

- Adds a new local viewer flow to the Python CLI and web tooling under `src/swim_pose/`.
- Reuses existing prediction JSONL outputs in `artifacts/predictions/` and optional report JSON files in `artifacts/reports/`.
- Affects developer workflow and README documentation by adding a visual inspection step after inference and evaluation.
