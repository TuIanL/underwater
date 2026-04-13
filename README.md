# Swim Pose

This repository contains the current implementation pass for breaststroke 2D keypoint localization on a single stitched video per clip. The current scope covers annotation tooling, dataset bookkeeping, a supervised baseline, and semi-supervised scaffolding for the stitched-video-first workflow.

## Layout

- `configs/`: example experiment and data configuration files
- `docs/`: annotation guide and project notes
- `data/manifests/`: clip manifests and split outputs
- `data/templates/`: annotation and manifest templates
- `reports/`: human-readable reports such as failure mode logs
- `src/swim_pose/`: implementation code and CLI entrypoints

## Quickstart

```bash
uv sync --locked
```

The documented workflow uses `uv run ...`, so you do not need to manually activate `.venv` first. Supported commands are expected to work from the repository root and from subdirectories inside the repository.

## Common Commands

```bash
uv run swim-pose manifest init --video-root data/raw/videos --output data/manifests/clips.csv
uv run swim-pose manifest audit --manifest data/manifests/clips.csv --output data/manifests/clips.audit.csv
uv run swim-pose frames extract --manifest data/manifests/clips.audit.csv --output-root data/frames --index-output data/manifests/unlabeled_index.csv --views stitched --every-nth 5
uv run swim-pose annotations template --output data/templates/frame_annotation.example.json
uv run swim-pose annotations validate --input data/templates/frame_annotation.example.json
uv run swim-pose seed select --manifest data/manifests/clips.audit.csv --output data/manifests/seed_frames.csv --source-view stitched
uv run swim-pose annotations scaffold --seed-csv data/manifests/seed_frames.csv --frame-root data/frames --output-root data/annotations/seed
uv run swim-pose annotations gui --annotation-root data/annotations/seed --frame-root data/frames
uv run swim-pose annotations audit --annotation-root data/annotations/seed --output reports/annotation-audit.json
uv run swim-pose dataset split --index data/manifests/annotation_index.csv --output-dir data/manifests/splits
uv run swim-pose train supervised --config configs/supervised.toml
uv run swim-pose predict --config configs/supervised.toml --checkpoint artifacts/checkpoints/supervised/best.pt --index data/manifests/splits/val.csv --output artifacts/predictions/val_predictions.jsonl
uv run swim-pose evaluate --predictions artifacts/predictions/val_predictions.jsonl --annotations data/manifests/splits/val.csv --output artifacts/reports/val_metrics.json
uv run swim-pose predictions web --predictions artifacts/predictions/supervised_labeled_predictions.jsonl --frame-root data/frames --report artifacts/reports/supervised_eval.json
uv run swim-pose pseudolabel generate --predictions artifacts/predictions/val_predictions.jsonl --output artifacts/predictions/pseudolabels.jsonl
uv run swim-pose train semisupervised --config configs/semi_supervised.toml
```

## Prediction Viewer

After `predict`, you can inspect one model's results in a local browser UI:

```bash
uv run swim-pose predictions web \
  --predictions artifacts/predictions/supervised_labeled_predictions.jsonl \
  --frame-root data/frames \
  --report artifacts/reports/supervised_eval.json
```

What the viewer shows:

- the original frame with the predicted 18-point skeleton overlay
- continuous playback controls for turning a prediction sequence into a local pose video-style review
- clip and frame navigation for browsing one prediction file
- per-keypoint coordinates, confidence, and visibility values
- optional overall and per-joint metrics when a report JSON is provided

If the prediction file was generated from sampled frames rather than every original video frame, playback will still be sequence playback, but it will look like a sampled replay instead of a true full-frame reconstruction.

If you only want qualitative inspection, `--report` is optional:

```bash
uv run swim-pose predictions web \
  --predictions artifacts/predictions/supervised_unlabeled_predictions.jsonl \
  --frame-root data/frames
```

If you want a pure clip-based player view that auto-plays and hides the long frame list:

```bash
uv run swim-pose predictions web \
  --predictions artifacts/predictions/supervised_unlabeled_predictions.jsonl \
  --frame-root data/frames \
  --player-mode
```

If one predictions file contains multiple clips, you can jump directly into one clip:

```bash
uv run swim-pose predictions web \
  --predictions artifacts/predictions/supervised_unlabeled_predictions.jsonl \
  --frame-root data/frames \
  --player-mode \
  --clip athlete01_session01_蛙
```

## Path Rules

- Relative config, checkpoint, manifest, report, and output paths are interpreted from the repository root.
- Frame references stored inside annotations and frame indices stay relative to the configured `frame_root` / `image_root`.
- Manifest source video paths are stored repository-relative when the videos live inside the repository, and absolute when they live outside it.
- If you have an older manifest with caller-relative paths such as `../videos/...`, migrate it before use:

```bash
uv run swim-pose manifest migrate-paths \
  --manifest data/manifests/clips.csv \
  --output data/manifests/clips.migrated.csv \
  --legacy-base /path/to/the/original/working-directory
```

## Annotation Contract

The canonical keypoint schema uses 18 landmarks:

`nose`, `neck`, `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist`, `left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_ankle`, `right_ankle`, `left_heel`, `right_heel`, `left_toe`, `right_toe`

Visibility states are:

- `2`: directly visible
- `1`: occluded but inferable
- `0`: not visible and not reliably inferable

See `docs/annotation-guide.md` for the detailed labeling rules.

## Raw Video Placement

Put baseline source videos under `data/raw/videos/<athlete_id>/<session_id>/`.

The current baseline expects one stitched video per clip:

```text
data/raw/videos/
  athlete01/
    session01/
      trial01_stitched.mp4
```

If you also keep separate raw camera files for provenance or future experiments, keep the same clip stem:

```text
data/raw/videos/
  athlete01/
    session01/
      trial01_above.mp4
      trial01_under.mp4
      trial01_stitched.mp4
```

Supported suffixes:

- `_above` for the above-water camera
- `_under` for the underwater camera
- `_stitched` for the baseline composite video

If a file has no view suffix, the tooling treats it as `stitched`.

## Manifest Notes

`stitched_path` is the required baseline field for extraction, annotation, training, inference, and evaluation. `raw_above_path`, `raw_under_path`, and the audit status columns are optional provenance metadata when separate camera files exist.

## Labeling Workflow

`swim-pose annotations gui --annotation-root data/annotations/seed --frame-root data/frames` means:

- `annotations gui`: open the local desktop labeling tool
- `--annotation-root data/annotations/seed`: read and save the JSON annotation files under `data/annotations/seed`
- `--frame-root data/frames`: load the corresponding JPG frame images from `data/frames`

Recommended workflow:

1. Run `uv run swim-pose annotations gui --annotation-root data/annotations/seed --frame-root data/frames`
2. For frames where the swimmer is visible, place the 18 keypoints and mark the frame as `Labeled`
3. For frames where the swimmer has not entered the view yet, use `No Swimmer` instead of silently skipping the file
4. If you are not finished with a frame, keep it as `Pending`; if it needs another pass, mark it as `Review`
5. Save and close the GUI
6. Run `uv run swim-pose annotations audit --annotation-root data/annotations/seed --output reports/annotation-audit.json`
7. If the audit looks clean, build the annotation index:

```bash
uv run swim-pose annotations index --annotation-root data/annotations/seed --output data/manifests/annotation_index.csv
```

GUI shortcuts:

- Left click: place the selected keypoint
- Right click / `Delete` / `Backspace` / `c`: clear the selected keypoint
- `Ctrl+Z`: undo the last edit
- `0`, `1`, `2`: set visibility
- `j`, `k`: switch keypoints
- `a`, `d`: previous or next annotation file
- `l`: mark `Labeled`
- `n`: mark `No Swimmer`
- `r`: mark `Review`
- `p`: mark `Pending`

Only frames with `frame_status = labeled` are written into `data/manifests/annotation_index.csv`. Frames marked `pending`, `review`, or `no_swimmer` stay out of the training index.

If the desktop Tk GUI crashes on macOS, use the browser-based annotation UI instead:

```bash
./scripts/launch_annotation_gui.sh
```

This wrapper now launches `annotations web` through the project's `uv` environment and opens the labeling page in your browser.
