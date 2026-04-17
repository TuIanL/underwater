## Why

Training a 3D or fully temporal swim-pose model from scratch remains too costly and risky for the current dataset scale, while the business goal is still reliable landmark trajectories for downstream biomechanical analysis. The repository's current failure modes are concentrated in underwater occlusion, stitched-waterline distortion, and frame-to-frame flicker, so a stronger pretrained 2D pose baseline with targeted underwater adaptation is a more practical path to usable results now.

## What Changes

- Adopt a YOLOv8-Pose-based 2D localization baseline as the default training and inference path for stitched swim video frames.
- Preserve the project's 18-point anatomical output contract while introducing an explicit mapping and extension strategy from general human-pose priors to the swim-specific schema.
- Add an underwater-domain adaptation workflow focused on splash, occlusion, seam artifacts, and swimmer-specific visual conditions rather than on full from-scratch spatiotemporal pretraining.
- Add confidence-aware temporal postprocessing so downstream biomechanics consumers can use stabilized trajectories without losing access to raw frame-level predictions.
- Reposition the existing SupCon and video-to-2D bridge work as optional research paths rather than the default localization architecture.
- Update evaluation and reporting so temporal stability metrics distinguish real swimmer motion from prediction flicker and compare raw versus filtered trajectories explicitly.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `breaststroke-keypoint-localization`: change the default localization baseline from a custom frame-level heatmap pipeline toward a pretrained YOLOv8-Pose 2D baseline with underwater fine-tuning, explicit 18-keypoint contract preservation, temporal postprocessing, and revised stability-reporting requirements.

## Impact

- Affected code: `src/swim_pose/training/`, `src/swim_pose/cli.py`, prediction export/evaluation utilities, config files under `configs/`, artifact packaging/reporting, and dataset-conversion utilities for pose training.
- Affected dependencies: adds an external YOLO pose training/runtime dependency and related dataset/config wiring.
- Affected workflows: supervised baseline training, inference export, evaluation, pseudo-label generation, and downstream biomechanics preprocessing will all need to distinguish raw and temporally filtered outputs.
