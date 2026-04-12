## MODIFIED Requirements

### Requirement: Canonical dual-view source data
The project SHALL treat one stitched composite video as the canonical source data for breaststroke keypoint localization. The baseline manifest MUST require the stitched clip path, athlete identifier, and session identifier for each clip. Raw above-water and underwater videos MAY be recorded as optional provenance metadata when available, but they MUST NOT be required inputs for baseline training or evaluation.

#### Scenario: Preparing a clip for training or evaluation
- **WHEN** a clip is prepared for the localization pipeline
- **THEN** the dataset manifest records the stitched clip path together with the athlete identifier and session identifier as the required baseline fields

#### Scenario: Raw camera files also exist for a clip
- **WHEN** a clip also has separate above-water or underwater source files
- **THEN** those files are treated as optional metadata and do not change the stitched clip's role as the canonical baseline input

### Requirement: Localization model outputs
The first-phase model SHALL output 2D coordinates, confidence values, and visibility-related predictions for the full 18-point schema on each processed stitched-video frame or stitched-frame window. Downstream consumers MUST be able to distinguish low-confidence predictions from absent or non-inferable keypoints.

#### Scenario: A stitched-video frame is processed at inference time
- **WHEN** the model returns keypoint predictions for a frame from the stitched clip
- **THEN** each of the 18 keypoints includes coordinates or an explicit absent state together with a confidence value that downstream logic can inspect

#### Scenario: A point is highly uncertain
- **WHEN** the model cannot localize a keypoint reliably because of occlusion, seam artifacts, or motion noise in the stitched video
- **THEN** the output preserves low confidence instead of forcing a high-certainty coordinate

### Requirement: Evaluation and reporting
The project SHALL report localization quality on stitched-video inputs using normalized keypoint error or PCK-style accuracy, per-joint metrics, visible-versus-occluded performance, temporal stability, and held-out athlete or held-out session results. Aggregate metrics alone MUST NOT be the only reported outcome.

#### Scenario: A model checkpoint is evaluated
- **WHEN** validation or test results are published for the stitched-video baseline
- **THEN** the report includes both overall metrics and joint-level breakdowns for difficult landmarks such as wrists, ankles, heels, and toes

#### Scenario: Generalization is assessed
- **WHEN** the team claims an improvement over a prior stitched-video baseline
- **THEN** the comparison is made on held-out athletes or sessions rather than on frame-randomized splits
