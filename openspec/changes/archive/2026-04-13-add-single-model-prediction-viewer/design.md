## Context

The repository already supports a full offline workflow for stitched-video localization: `swim-pose predict` exports frame-level JSONL predictions, `swim-pose evaluate` exports report JSON, and `annotation_web.py` provides a browser-based local UI for annotation review. What is missing is a read-only inspection surface for model outputs. Right now, understanding prediction quality requires opening raw JSONL files and mentally mapping coordinates back to images, which is slow and makes it hard to communicate what the model is actually doing.

The main constraints are:
- keep the workflow local-first and lightweight, matching the existing Python CLI tooling
- avoid adding a heavy frontend toolchain or new server dependency for a simple inspection page
- reuse the current prediction JSONL and evaluation JSON formats rather than inventing a second export format
- support browsing across many predicted frames without requiring manual file editing

## Goals / Non-Goals

**Goals:**
- Add a single-model, read-only browser UI for inspecting prediction results on top of frame images.
- Reuse the existing local web-server pattern already established by the annotation tool.
- Support frame browsing by clip and frame index, with visible skeleton overlay, per-point metadata, and optional report metrics.
- Keep the first version aligned with existing artifact formats in `artifacts/predictions/` and `artifacts/reports/`.

**Non-Goals:**
- Multi-model comparison or metric diffing between checkpoints.
- Editing predictions, correcting annotations, or writing new labels from the viewer.
- Training, inference, or evaluation execution from inside the browser UI.
- Remote deployment, authentication, or multi-user collaboration.

## Decisions

### 1. Add a dedicated read-only prediction viewer module

The change will introduce a new local web module under `src/swim_pose/` rather than extending the annotation UI in-place. The annotation tool is centered on mutable annotation JSON files and save workflows, while the prediction viewer is a read-only inspector over inference outputs and optional report files. Keeping these concerns separate makes the HTTP routes, UI copy, and data flow easier to reason about.

Alternatives considered:
- Extend `annotation_web.py` with a read-only prediction mode: rejected because it would mix editable annotation state with immutable prediction artifacts and make the page harder to maintain.
- Build a separate frontend app with a JS framework: rejected because the project already has a viable local-web pattern and does not need a full frontend toolchain for this scope.

### 2. Reuse the existing local HTTP + inline HTML/CSS/JS pattern

The viewer will follow the same runtime shape as the annotation web UI: a Python stdlib HTTP server serving one page, JSON endpoints, and frame image bytes. This keeps the feature install-free beyond existing project dependencies and preserves the current local-desktop workflow.

Alternatives considered:
- Introduce Flask/FastAPI: unnecessary extra dependency and operational surface for a tool that is only used locally.
- Generate static HTML after prediction: rejected because the viewer still needs local image resolution and interactive frame navigation.

### 3. Introduce a CLI namespace for viewing prediction artifacts

The CLI will expose a dedicated viewer entrypoint separate from `predict`, because inference generation and result inspection are different tasks. A noun-based namespace such as `swim-pose predictions web` keeps room for future viewer-related subcommands while staying consistent with the existing `annotations web` pattern.

Expected arguments:
- `--predictions`: required JSONL file
- `--frame-root`: required frame root used to resolve relative `image_path` values
- `--report`: optional evaluation JSON
- `--host`, `--port`, `--no-browser`: local serving controls matching the annotation tool

Alternatives considered:
- Add flags onto `swim-pose predict`: rejected because it would overload one command with both generation and viewing concerns.
- Add a generic `viewer` top-level command: workable, but less descriptive if more viewers are added later.

### 4. Load predictions once at startup and serve indexed read models

The viewer will parse the JSONL file when the process starts, build an ordered frame list, and expose read endpoints backed by in-memory metadata. This makes frame navigation and filtering responsive and keeps the browser code simple. The current project scale and artifact sizes make this a reasonable first trade-off.

The in-memory read model should include:
- ordered items with `clip_id`, `frame_index`, `source_view`, and `image_path`
- current-frame prediction payload for overlay and detail rendering
- unique clip identifiers for UI filtering
- optional report payload loaded once if `--report` is supplied

Alternatives considered:
- Re-read the JSONL file on every request: simpler backend state, but poor local responsiveness and repeated parsing cost.
- Build a database or cache layer: too heavy for a first offline inspection tool.

### 5. Keep the UI focused on one inspection loop: select frame, inspect overlay, inspect details

The page should optimize for quick qualitative review. The main canvas shows the selected image with skeleton overlay. Supporting panels show frame metadata, clip/frame selection, per-keypoint confidence and visibility, and optional report metrics. A confidence threshold control should let users hide or de-emphasize uncertain points without changing the underlying data.

Recommended first-layout structure:
- main viewer area: frame image with predicted skeleton overlay
- navigation/filter panel: clip selector, frame list or sequential navigation controls
- detail panel: current frame metadata plus per-keypoint table
- summary area: overall metrics and per-joint report values when a report JSON is available

Alternatives considered:
- A dashboard-first metrics page with frame viewer as a secondary detail: rejected because the user's primary need is to see what the model predicted on images.
- A pure frame-by-frame player without keypoint tables: rejected because confidence and visibility values are part of the model contract and need to be inspectable.

### 6. Treat missing optional data as non-fatal, but fail clearly on required inputs

The viewer must require a readable prediction JSONL file and resolvable frame root. Optional report data should enrich the page when available, but absence of a report must not block frame inspection. Missing frame files or malformed prediction rows should surface as explicit UI or CLI errors rather than silently failing.

Alternatives considered:
- Make report JSON required: rejected because qualitative frame inspection is still valuable without aggregate metrics.
- Ignore missing assets and skip frames silently: rejected because silent gaps make debugging data issues harder.

## Risks / Trade-offs

- [Large prediction files could consume noticeable memory at startup] -> Accept eager loading for v1, and revisit lazy paging only if artifact sizes make startup painful in practice.
- [A separate prediction viewer duplicates some web-server scaffolding from the annotation tool] -> Keep module structure parallel and straightforward for now; extract shared helpers later only if both UIs start changing together.
- [Confidence thresholding can hide real failures if misused] -> Default to showing all points initially and make any filtering state explicit in the UI.
- [Frame browsing across many clips could still feel heavy without stronger search tools] -> Include clip-level filtering in the first version so users can narrow the frame list before deeper navigation.

## Migration Plan

1. Add the new viewer module and CLI entrypoint without changing existing prediction or evaluation formats.
2. Document the post-inference workflow in the README so users can launch the viewer immediately after `predict` and `evaluate`.
3. Validate the viewer against existing artifacts such as `artifacts/predictions/supervised_unlabeled_predictions.jsonl` and `artifacts/reports/supervised_eval.json`.
4. Keep the feature optional and local-only so rollback is simply not using the new command if problems are found.

## Open Questions

- Should the first version auto-discover a likely report JSON near the predictions file, or require an explicit `--report` argument for predictability?
- Is a simple clip dropdown enough for large files, or do we expect to need text search and frame-range jumping immediately?
- Do we want to show invisible points with a different visual treatment, or hide them by default once the viewer exists?
