## Why

The project needs a clear first milestone for swim posture analysis that is technically achievable with limited labeling resources and professionally captured data. Breaststroke keypoint localization is the right starting point because downstream error detection and quality scoring depend on stable, anatomically meaningful joint positions under waterline transitions, reflections, bubbles, and self-occlusion.

## What Changes

- Define a new dual-view breaststroke keypoint localization capability focused on precise 2D joint detection before any error diagnosis or scoring.
- Standardize an 18-point keypoint schema for breaststroke analysis, including head, upper-limb, lower-limb, heel, and toe landmarks.
- Standardize annotation rules for visibility, occlusion, frame sampling, athlete-based dataset splits, and difficult-frame coverage.
- Define a training approach based on pretrained human pose initialization, small high-quality manual labels, and semi-supervised learning on large unlabeled breaststroke videos.
- Define evaluation expectations for point accuracy, occluded-point performance, temporal stability, and split-by-athlete generalization.

## Capabilities

### New Capabilities
- `breaststroke-keypoint-localization`: Defines the annotation contract, training scope, model outputs, and evaluation criteria for dual-view breaststroke 2D keypoint localization.

### Modified Capabilities
- None.

## Impact

- Affects future data collection, annotation workflow, model training pipeline, and evaluation reporting.
- Establishes the project contract for dual-view video usage, keypoint output semantics, and semi-supervised training assumptions.
- Defers stroke error detection, coaching feedback, and quality scoring to later changes built on top of this localization baseline.
