# prediction-viewer Specification

## Purpose
TBD - created by archiving change add-single-model-prediction-viewer. Update Purpose after archive.
## Requirements
### Requirement: Local prediction viewer launch
The system SHALL provide a local browser-based viewer for one prediction JSONL artifact at a time. Users MUST be able to launch the viewer from the CLI by supplying a prediction file and frame root, with an optional evaluation report JSON that enriches the UI when present.

#### Scenario: Launching the viewer with required inputs
- **WHEN** a user runs the prediction viewer command with a readable predictions JSONL file and a valid frame root
- **THEN** the system starts a local HTTP session and serves a browser UI for inspecting that prediction set

#### Scenario: Launching the viewer with an optional report file
- **WHEN** a user also supplies a readable evaluation report JSON
- **THEN** the served UI includes the report data in addition to frame-level prediction browsing

### Requirement: Frame browsing and overlay inspection
The viewer SHALL let the user browse predicted frames from the loaded artifact and inspect each selected frame as an image overlaid with the model's 18-point skeleton output. The browsing experience MUST expose enough frame metadata to distinguish clips, source views, and frame order.

#### Scenario: Selecting a predicted frame
- **WHEN** a user selects a frame entry from the viewer
- **THEN** the UI loads the corresponding image and draws the predicted keypoints and skeleton edges for that frame

#### Scenario: Narrowing inspection to a clip
- **WHEN** the loaded prediction artifact contains frames from multiple clips
- **THEN** the UI allows the user to filter or group frames by `clip_id` so they can inspect one clip without stepping through the entire file sequentially

### Requirement: Per-keypoint prediction details
For the currently selected frame, the viewer SHALL display each predicted keypoint's coordinates, confidence value, and visibility prediction. The UI MUST provide a way to visually de-emphasize or hide low-confidence predictions without altering the underlying artifact data.

#### Scenario: Inspecting current-frame keypoint details
- **WHEN** a frame is open in the viewer
- **THEN** the UI shows a per-keypoint breakdown for the full 18-point schema, including confidence and visibility values for each point

#### Scenario: Adjusting the confidence filter
- **WHEN** a user changes the confidence threshold control
- **THEN** the overlay updates to reflect the chosen threshold while preserving the original prediction values in the detail panel

### Requirement: Optional report metrics summary
When an evaluation report JSON is available, the viewer SHALL display both overall metrics and per-joint metrics alongside the frame viewer. The absence of a report MUST NOT prevent users from browsing frame-level predictions.

#### Scenario: Report metrics are provided
- **WHEN** the viewer is launched with an evaluation report JSON
- **THEN** the UI shows overall summary metrics and per-joint metric entries from that report

#### Scenario: Report metrics are not provided
- **WHEN** the viewer is launched without an evaluation report JSON
- **THEN** the UI still allows full frame browsing and clearly indicates that aggregate metrics are unavailable for the current session

### Requirement: Clear handling of local asset and data issues
The viewer SHALL fail clearly when required inputs cannot be read, and it MUST surface missing frame assets or malformed prediction records as explicit errors rather than silently omitting them from the session.

#### Scenario: Predictions file cannot be loaded
- **WHEN** the user launches the viewer with a missing or unreadable predictions JSONL path
- **THEN** the command exits with a clear error instead of starting a partial session

#### Scenario: A selected frame image cannot be resolved
- **WHEN** the viewer tries to load a frame image whose `image_path` does not resolve under the configured frame root
- **THEN** the session reports that frame-loading error explicitly so the user can diagnose the artifact or path mismatch

