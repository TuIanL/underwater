## 1. Data Inventory And Annotation Setup

- [x] 1.1 Build a dataset manifest that links each clip to the raw above-water stream, raw underwater stream, stitched review clip, athlete ID, and session ID
- [x] 1.2 Verify synchronization quality for each dual-view clip and mark stitched-only or poorly aligned clips as degraded inputs
- [x] 1.3 Publish the 18-point keypoint guide with anatomical left/right rules, heel/toe definitions, and visibility state examples
- [x] 1.4 Configure the annotation format and tooling so every frame stores all 18 keypoints plus visibility states `0/1/2`

## 2. Seed Dataset Creation

- [x] 2.1 Select seed annotation frames across breaststroke cycle phases, athletes, sessions, waterline crossings, and other hard conditions
- [x] 2.2 Annotate and review the first high-quality seed set, including difficult frames with splash, overlap, and partial out-of-frame joints
- [x] 2.3 Audit a review subset for left/right consistency, heel/toe correctness, and visibility-state correctness before freezing the seed set
- [x] 2.4 Create train, validation, and test partitions grouped by athlete or session to avoid frame-level leakage

## 3. Baseline Localization Pipeline

- [x] 3.1 Prepare a baseline 2D keypoint training pipeline initialized from a pretrained human pose model
- [x] 3.2 Implement supervised training on the reviewed seed set and export predictions with coordinates, confidence, and visibility-related outputs
- [x] 3.3 Add baseline evaluation for normalized keypoint error or PCK-style metrics, per-joint breakdowns, and visible-versus-occluded performance
- [ ] 3.4 Measure temporal stability on held-out clips and record failure cases around waterline and occlusion-heavy frames

## 4. Semi-Supervised Expansion

- [x] 4.1 Generate pseudo-labels on unlabeled breaststroke clips using confidence thresholds from the supervised baseline
- [x] 4.2 Add semi-supervised training support for confidence-gated pseudo-labels and augmentation-consistency losses
- [x] 4.3 Experiment with temporal consistency or smoothing constraints after the single-frame baseline is stable
- [ ] 4.4 Compare supervised and semi-supervised results on held-out athletes and sessions using the same reporting protocol

## 5. Readiness For Downstream Analysis

- [x] 5.1 Package the best checkpoint, schema definition, and evaluation report as the project's localization baseline
- [x] 5.2 Document known failure modes and open questions that must be resolved before adding stroke error detection or quality scoring
