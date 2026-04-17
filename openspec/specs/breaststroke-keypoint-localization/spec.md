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
The first-phase localization system SHALL output frame-keyed raw 2D coordinates, confidence values, and visibility-related predictions for the full 18-point schema on each processed stitched-video frame. The default baseline MAY use an external pretrained pose architecture internally, but inference and exported predictions MUST preserve the project schema and one-target-frame contract. When temporal postprocessing is enabled, the system SHALL additionally emit filtered trajectories keyed to the same frames while keeping raw predictions inspectable. Downstream consumers MUST be able to distinguish low-confidence raw predictions, absent or non-inferable keypoints, and filtered trajectory values.

#### Scenario: A stitched-video frame is processed by the default baseline
- **WHEN** the model returns keypoint predictions for a frame from the stitched clip
- **THEN** each of the 18 project keypoints includes raw coordinates or an explicit absent state together with a confidence value that downstream logic can inspect

#### Scenario: A point is highly uncertain
- **WHEN** the model cannot localize a keypoint reliably because of occlusion, seam artifacts, or motion noise in the stitched video
- **THEN** the raw output preserves low confidence instead of forcing a high-certainty coordinate

#### Scenario: Filtered trajectories are exported
- **WHEN** temporal postprocessing is enabled during inference or export
- **THEN** the resulting artifacts remain frame-keyed and let downstream tooling tell raw predictions apart from filtered trajectory values for the same frame

### Requirement: Semi-supervised training workflow
The training pipeline SHALL use a pretrained 2D human-pose initialization as the default localization baseline and SHALL support underwater-domain fine-tuning on stitched swim frames with augmentations, sampling, or curricula targeted at splash, blur, seam artifacts, and self-occlusion. The pipeline SHALL continue to support expansion from a manually reviewed seed set to larger unlabeled breaststroke corpora through pseudo-labeling or consistency-based semi-supervised learning, but those extensions MUST remain downstream of a working supervised baseline and MUST record the confidence criteria used to accept, reject, or down-weight unlabeled targets. The localization baseline MUST NOT depend on training a 3D or fully temporal model from scratch before usable 2D predictions are available.

#### Scenario: Establishing the first baseline
- **WHEN** the project starts model training under the new default architecture
- **THEN** the initial supervised baseline is trained from a pretrained 2D human-pose model on manually reviewed seed annotations rather than on unlabeled discovery alone

#### Scenario: Fine-tuning for underwater conditions
- **WHEN** the team adapts the default baseline to stitched swim footage
- **THEN** the training recipe explicitly targets underwater and seam-related failure modes instead of assuming generic above-water pose priors are sufficient

#### Scenario: Adding unlabeled videos
- **WHEN** pseudo-labels or consistency targets are generated for unlabeled breaststroke clips
- **THEN** the pipeline records the confidence criteria used to accept, reject, or down-weight those targets and compares the result against the supervised baseline

### Requirement: Evaluation and reporting
The project SHALL report localization quality on stitched-video inputs using normalized keypoint error or PCK-style accuracy, per-joint metrics, visible-versus-occluded performance, held-out athlete or held-out session results, and temporal stability reporting for both raw and filtered trajectories when filtering is enabled. Temporal stability reporting MUST distinguish real swimmer motion from prediction flicker by using residual-, consistency-, or plausibility-oriented measures in addition to simple frame-to-frame displacement. Aggregate metrics alone MUST NOT be the only reported outcome.

#### Scenario: A model checkpoint is evaluated
- **WHEN** validation or test results are published for the stitched-video baseline
- **THEN** the report includes overall metrics, joint-level breakdowns for difficult landmarks such as wrists, ankles, heels, and toes, and raw-versus-filtered comparisons when filtered outputs exist

#### Scenario: A temporal filter is claimed to improve stability
- **WHEN** the team claims an improvement from temporal postprocessing
- **THEN** the comparison reports raw and filtered metrics on the same held-out split so the filter does not hide a degraded base detector

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

### Requirement: Pretrained baseline schema preservation
When the localization baseline is initialized from an external pretrained pose model, the system SHALL preserve the project's full 18-keypoint breaststroke schema and SHALL define an explicit mapping or extension strategy from the upstream pose schema into the project schema. The training and inference pipeline MUST NOT silently drop project-required landmarks such as `neck`, `heel`, or `toe`, and it MUST preserve anatomical left/right semantics across all adapted outputs.

#### Scenario: Upstream pretrained model uses a smaller or different keypoint schema
- **WHEN** the team adopts a pretrained pose model whose native keypoints do not exactly match the project's 18-keypoint schema
- **THEN** the pipeline records an explicit schema-adaptation strategy and still trains and exports all 18 required project keypoints

#### Scenario: A project-specific landmark remains uncertain after adaptation
- **WHEN** a landmark that is not directly covered by the upstream pretrained schema cannot be localized reliably in a frame
- **THEN** the exported prediction preserves low confidence or an absent state rather than fabricating a high-certainty coordinate

### Requirement: Temporal trajectory postprocessing
The localization workflow SHALL provide a confidence-aware temporal postprocessing stage that derives stabilized joint trajectories from frame-keyed raw pose predictions for downstream biomechanics analysis. The postprocessing stage MUST accept raw confidence or visibility signals as inputs, MUST be explicitly disable-able, and MUST emit filtered trajectories without overwriting the corresponding raw predictions.

#### Scenario: Temporal postprocessing is enabled for biomechanics export
- **WHEN** a prediction run enables temporal postprocessing
- **THEN** the exported artifacts include both raw frame-keyed predictions and filtered trajectories keyed to the same source frames

#### Scenario: A keypoint becomes unreliable across adjacent frames
- **WHEN** the raw detector confidence collapses or the keypoint becomes non-inferable during a short temporal span
- **THEN** the postprocessing stage down-weights or propagates the uncertainty instead of inventing a stable high-confidence motion path

