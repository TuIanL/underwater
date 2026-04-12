# Swim Pose

This repository contains the first implementation pass for the `add-breaststroke-keypoint-localization` OpenSpec change. The current scope is dual-view breaststroke 2D keypoint localization with annotation tooling, dataset bookkeeping, a supervised baseline, and semi-supervised scaffolding.

## Layout

- `configs/`: example experiment and data configuration files
- `docs/`: annotation guide and project notes
- `data/manifests/`: clip manifests and split outputs
- `data/templates/`: annotation and manifest templates
- `reports/`: human-readable reports such as failure mode logs
- `src/swim_pose/`: implementation code and CLI entrypoints

## Quickstart

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Common Commands

```bash
swim-pose manifest init --video-root data/raw/videos --output data/manifests/clips.csv
swim-pose manifest audit-sync --manifest data/manifests/clips.csv --output data/manifests/clips.audit.csv
swim-pose frames extract --manifest data/manifests/clips.audit.csv --output-root data/frames --index-output data/manifests/unlabeled_index.csv --views stitched --every-nth 5
swim-pose annotations template --output data/templates/frame_annotation.example.json
swim-pose annotations validate --input data/templates/frame_annotation.example.json
swim-pose seed select --manifest data/manifests/clips.audit.csv --output data/manifests/seed_frames.csv --source-view stitched
swim-pose annotations scaffold --seed-csv data/manifests/seed_frames.csv --frame-root data/frames --output-root data/annotations/seed
swim-pose annotations gui --annotation-root data/annotations/seed --frame-root data/frames
swim-pose annotations audit --annotation-root data/annotations/seed --output reports/annotation-audit.json
swim-pose dataset split --index data/manifests/annotation_index.csv --output-dir data/manifests/splits
swim-pose train supervised --config configs/supervised.toml
swim-pose predict --config configs/supervised.toml --checkpoint artifacts/checkpoints/supervised/best.pt --index data/manifests/splits/val.csv --output artifacts/predictions/val_predictions.jsonl
swim-pose evaluate --predictions artifacts/predictions/val_predictions.jsonl --annotations data/manifests/splits/val.csv --output artifacts/reports/val_metrics.json
swim-pose pseudolabel generate --predictions artifacts/predictions/val_predictions.jsonl --output artifacts/predictions/pseudolabels.jsonl
swim-pose train semisupervised --config configs/semi_supervised.toml
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

Put untouched source videos under `data/raw/videos/<athlete_id>/<session_id>/`.

If you only have stitched videos, this is fine for the first baseline:

```text
data/raw/videos/
  athlete01/
    session01/
      trial01_stitched.mp4
```

If you later collect separate raw views, keep the same clip stem for synchronized files:

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
- `_stitched` for the preview/composite video if you have it

If a file has no view suffix, the tooling treats it as `stitched`.

## Labeling Workflow

`swim-pose annotations gui --annotation-root data/annotations/seed --frame-root data/frames` means:

- `annotations gui`: open the local desktop labeling tool
- `--annotation-root data/annotations/seed`: read and save the JSON annotation files under `data/annotations/seed`
- `--frame-root data/frames`: load the corresponding JPG frame images from `data/frames`

Recommended workflow:

1. Run `swim-pose annotations gui --annotation-root data/annotations/seed --frame-root data/frames`
2. For frames where the swimmer is visible, place the 18 keypoints and mark the frame as `Labeled`
3. For frames where the swimmer has not entered the view yet, use `No Swimmer` instead of silently skipping the file
4. If you are not finished with a frame, keep it as `Pending`; if it needs another pass, mark it as `Review`
5. Save and close the GUI
6. Run `swim-pose annotations audit --annotation-root data/annotations/seed --output reports/annotation-audit.json`
7. If the audit looks clean, build the annotation index:

```bash
swim-pose annotations index --annotation-root data/annotations/seed --output data/manifests/annotation_index.csv
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

This script now starts `annotations web` with the system Python and opens the labeling page in your browser.
