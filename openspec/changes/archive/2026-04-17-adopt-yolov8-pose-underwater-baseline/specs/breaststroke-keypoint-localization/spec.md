## ADDED Requirements

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

## MODIFIED Requirements

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
