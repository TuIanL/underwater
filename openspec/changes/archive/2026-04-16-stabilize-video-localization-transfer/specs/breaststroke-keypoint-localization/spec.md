## ADDED Requirements

### Requirement: Dense-frame bridge context for video-to-2D transfer
When the localization pipeline enables video-to-2D bridge supervision, the temporal clip provided to the video teacher SHALL be assembled from dense frame context aligned to the labeled center frame's `clip_id`, `source_view`, and `frame_index`. The system MUST NOT satisfy bridge supervision by simply repeating the labeled frame or by approximating temporal context from sparse annotation rows alone. If valid dense context cannot be assembled for a sample, the bridge objective for that sample MUST be skipped or disabled explicitly.

#### Scenario: Dense temporal context exists for a labeled frame
- **WHEN** bridge-enabled supervised training requests a teacher clip for a labeled center frame
- **THEN** the dataset returns true neighboring frames from the same clip stream around that frame together with the center-frame supervision target

#### Scenario: Dense temporal context is unavailable
- **WHEN** the pipeline cannot find a valid dense clip for a labeled center frame
- **THEN** it disables or skips the bridge loss for that sample instead of repeating the center frame to fabricate a temporal clip

## MODIFIED Requirements

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
