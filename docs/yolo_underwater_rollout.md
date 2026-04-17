# YOLO Underwater Rollout

This note records the first side-by-side smoke comparison for the default YOLO underwater baseline and the legacy heatmap baseline.

## Scope

- Comparison date: 2026-04-17
- Data scope: current `annotation_index.csv`
- Labeled data size: 6 stitched frames from one athlete/session
- Purpose: validate the new training, export, and evaluation path end to end

## Commands Used

Legacy smoke config:

```bash
uv run -- swim-pose train supervised --config /tmp/swim_pose_compare_legacy.toml
uv run -- swim-pose predict \
  --config /tmp/swim_pose_compare_legacy.toml \
  --checkpoint /tmp/swim_pose_compare/legacy/best.pt \
  --index data/manifests/annotation_index.csv \
  --output /tmp/swim_pose_compare/legacy_predictions.jsonl
uv run -- swim-pose evaluate \
  --predictions /tmp/swim_pose_compare/legacy_predictions.jsonl \
  --annotations data/manifests/annotation_index.csv \
  --output /tmp/swim_pose_compare/legacy_report.json
```

YOLO smoke config:

```bash
uv run -- swim-pose train supervised --config /tmp/swim_pose_compare_yolo.toml
uv run -- swim-pose predict \
  --config /tmp/swim_pose_compare_yolo.toml \
  --checkpoint /tmp/swim_pose_compare/yolo/best.pt \
  --index data/manifests/annotation_index.csv \
  --output /tmp/swim_pose_compare/yolo_predictions.jsonl
uv run -- swim-pose evaluate \
  --predictions /tmp/swim_pose_compare/yolo_predictions.jsonl \
  --annotations data/manifests/annotation_index.csv \
  --output /tmp/swim_pose_compare/yolo_report.json
```

## Recorded Artifacts

- `artifacts/reports/yolo_underwater_smoke_comparison.json`
- `artifacts/reports/yolo_underwater_smoke_legacy_report.json`
- `artifacts/reports/yolo_underwater_smoke_yolo_report.json`

## Smoke Results

- Legacy raw `mean_normalized_error`: `0.4156`
- Legacy raw `pck@0.05`: `0.0404`
- Legacy raw `temporal_jitter`: `486.99`
- YOLO raw `mean_normalized_error`: `0.3694`
- YOLO raw `pck@0.05`: `0.0556`
- YOLO raw `temporal_jitter`: `0.0`

## Interpretation

- The new YOLO path trains, exports project-owned checkpoints, emits 18-keypoint frame-keyed predictions, and evaluates successfully on the current stitched-frame dataset.
- The smoke comparison is not promotion-ready evidence.
- The current labeled set is too small, and the YOLO smoke run produced scored predictions on only a subset of frames, so the apparent gains cannot be treated as a stable benchmark.

## Rollout Criteria

- Promote the YOLO baseline only after it is compared against the legacy heatmap path on held-out athlete or held-out session data.
- Require raw and filtered reports side by side for every promotion candidate.
- Require explicit review of project-specific landmarks: `neck`, `heel`, and `toe`.
- Require visual spot checks on splash-heavy and seam-crossing frames before updating the default training recommendation.

## Rollback Criteria

- Roll back to `configs/supervised_legacy.toml` if the YOLO raw detector materially regresses visible-joint accuracy on held-out data.
- Roll back postprocessing defaults if filtered trajectories hide detector failures or oversmooth kick dynamics.
- Keep bridge and SupCon configs in research mode until the YOLO baseline is stable enough that those comparisons are meaningful again.
