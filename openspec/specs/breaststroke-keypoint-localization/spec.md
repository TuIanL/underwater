# breaststroke-keypoint-localization Specification

## Purpose
TBD - created by archiving change add-breaststroke-keypoint-localization. Update Purpose after archive.
## Requirements
### Requirement: Canonical dual-view source data
The project SHALL treat one stitched composite video as the canonical source data for breaststroke keypoint localization. The baseline manifest MUST require the stitched clip path, athlete identifier, and session identifier for each clip. Raw above-water and underwater videos MAY be recorded as optional provenance metadata when available, but they MUST NOT be required inputs for baseline training or evaluation.

Optional transfer from Phase 1 video pretraining into the 2D localization pipeline MUST use an explicit bridge mechanism or bridge artifact. The localization baseline MUST NOT assume that a Phase 1 video-pretraining checkpoint is directly compatible with the 2D heatmap model.

#### Scenario: Preparing a clip for training or evaluation
- **WHEN** a clip is prepared for the localization pipeline
- **THEN** the dataset manifest records the stitched clip path together with the athlete identifier and session identifier as the required baseline fields

#### Scenario: Raw camera files also exist for a clip
- **WHEN** a clip also has separate above-water or underwater source files
- **THEN** those files are treated as optional metadata and do not change the stitched clip's role as the canonical baseline input

#### Scenario: Localization training wants to reuse Phase 1 information
- **WHEN** the team opts to transfer knowledge from a Phase 1 video-pretraining checkpoint into the 2D localization path
- **THEN** it does so through an explicit bridge strategy or bridge artifact rather than direct checkpoint loading

### Requirement: Breaststroke keypoint schema
The localization capability SHALL use exactly 18 keypoints for the first phase: `nose`, `neck`, `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist`, `left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_ankle`, `right_ankle`, `left_heel`, `right_heel`, `left_toe`, and `right_toe`. Left and right MUST refer to the athlete's anatomical left and right, not image left and right.

#### Scenario: A new annotation file is created
- **WHEN** an annotator starts labeling a frame
- **THEN** the file schema exposes all 18 required keypoints using the canonical names and anatomical left/right semantics

#### Scenario: Limbs overlap in the side view
- **WHEN** the swimmer's near-side and far-side limbs cross or overlap in the image
- **THEN** the annotation still preserves the athlete's anatomical left/right identities rather than swapping labels to match image order

### Requirement: Visibility-aware annotation semantics
Each annotated keypoint SHALL carry a visibility state with the values `2` for directly visible, `1` for occluded but inferable, and `0` for not visible and not reliably inferable. The annotation workflow MUST allow coordinates for state `1` and MUST allow null or omitted coordinates for state `0` according to the chosen storage format.

#### Scenario: A wrist is hidden behind splash but its position can be inferred
- **WHEN** the annotator can estimate the joint location from anatomy and nearby context
- **THEN** the wrist receives coordinates and visibility state `1`

#### Scenario: A toe leaves the frame entirely
- **WHEN** the keypoint is outside the image or fully indeterminate
- **THEN** the keypoint receives visibility state `0` and is not treated as a reliable supervised target

### Requirement: Annotation sampling and partitioning
The first labeled dataset SHALL sample frames across all major breaststroke cycle phases, multiple athletes, and difficult conditions such as waterline crossings, bubbles, reflections, and heavy self-occlusion. Train, validation, and test partitions MUST be split by athlete or capture session rather than by random frame assignment.

#### Scenario: Building the seed annotation set
- **WHEN** frames are selected for manual labeling
- **THEN** the selection includes both typical cycle frames and deliberately chosen hard frames instead of only dense adjacent frames from the same sequence

#### Scenario: Creating validation and test data
- **WHEN** dataset partitions are generated
- **THEN** no athlete-session group is split across train, validation, and test sets in a way that would leak near-duplicate frames

### Requirement: Annotation quality control
The annotation process SHALL include documented labeling guidance for each keypoint definition and a review process for difficult samples. The project MUST audit a subset of labeled frames for left/right consistency, heel/toe correctness, and visibility-state correctness before those labels are treated as seed supervision.

#### Scenario: A difficult frame is reviewed
- **WHEN** a frame contains heavy splash, waterline distortion, or overlapping limbs
- **THEN** the review process checks that keypoint identities and visibility states match the written annotation guidance

#### Scenario: Systematic label confusion is discovered
- **WHEN** quality checks find repeated mistakes such as swapped feet or inconsistent occlusion labels
- **THEN** the affected frames are corrected and the guidance is updated before continued large-scale labeling

### Requirement: Localization model outputs
The first-phase model SHALL output 2D coordinates, confidence values, and visibility-related predictions for the full 18-point schema on each processed stitched-video frame or stitched-frame window. When conservative video-to-2D transfer is enabled during training, inference and exported predictions MUST remain keyed to one target frame rather than a multi-frame prediction volume. Downstream consumers MUST be able to distinguish low-confidence predictions from absent or non-inferable keypoints.

#### Scenario: A stitched-video frame is processed at inference time
- **WHEN** the model returns keypoint predictions for a frame from the stitched clip
- **THEN** each of the 18 keypoints includes coordinates or an explicit absent state together with a confidence value that downstream logic can inspect, even if the checkpoint was trained with bridge supervision

#### Scenario: A point is highly uncertain
- **WHEN** the model cannot localize a keypoint reliably because of occlusion, seam artifacts, or motion noise in the stitched video
- **THEN** the output preserves low confidence instead of forcing a high-certainty coordinate

#### Scenario: Pseudo-label export consumes a bridge-trained checkpoint
- **WHEN** predictions from a bridge-trained localization checkpoint are exported for pseudo-label generation
- **THEN** the exported rows remain frame-keyed and compatible with the existing pseudo-label filtering workflow

### Requirement: Semi-supervised training workflow
The training pipeline SHALL use a human pose pretrained initialization and SHALL support expansion from a manually labeled seed set to larger unlabeled breaststroke corpora through pseudo-labeling or consistency-based semi-supervised learning. Unlabeled samples MUST be filtered or weighted by confidence so that low-quality pseudo-labels do not silently dominate training.

#### Scenario: Establishing the first baseline
- **WHEN** the project starts model training
- **THEN** the initial supervised baseline is trained on manually reviewed seed annotations rather than on unlabeled discovery alone

#### Scenario: Adding unlabeled videos
- **WHEN** pseudo-labels or consistency targets are generated for unlabeled breaststroke clips
- **THEN** the pipeline records the confidence criteria used to accept, reject, or down-weight those targets

### Requirement: Evaluation and reporting
The project SHALL report localization quality on stitched-video inputs using normalized keypoint error or PCK-style accuracy, per-joint metrics, visible-versus-occluded performance, temporal stability, and held-out athlete or held-out session results. Aggregate metrics alone MUST NOT be the only reported outcome.

#### Scenario: A model checkpoint is evaluated
- **WHEN** validation or test results are published for the stitched-video baseline
- **THEN** the report includes both overall metrics and joint-level breakdowns for difficult landmarks such as wrists, ankles, heels, and toes

#### Scenario: Generalization is assessed
- **WHEN** the team claims an improvement over a prior stitched-video baseline
- **THEN** the comparison is made on held-out athletes or sessions rather than on frame-randomized splits

### Requirement: Dense-frame bridge context for video-to-2D transfer
When the localization pipeline enables video-to-2D bridge supervision, the temporal clip provided to the video teacher SHALL be assembled from dense frame context aligned to the labeled center frame's `clip_id`, `source_view`, and `frame_index`. The system MUST NOT satisfy bridge supervision by simply repeating the labeled frame or by approximating temporal context from sparse annotation rows alone. If valid dense context cannot be assembled for a sample, the bridge objective for that sample MUST be skipped or disabled explicitly.

#### Scenario: Dense temporal context exists for a labeled frame
- **WHEN** bridge-enabled supervised training requests a teacher clip for a labeled center frame
- **THEN** the dataset returns true neighboring frames from the same clip stream around that frame together with the center-frame supervision target

#### Scenario: Dense temporal context is unavailable
- **WHEN** the pipeline cannot find a valid dense clip for a labeled center frame
- **THEN** it disables or skips the bridge loss for that sample instead of repeating the center frame to fabricate a temporal clip

