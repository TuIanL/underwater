## Why

The current swim pose specification still describes breaststroke localization as a dual-view pipeline built from synchronized above-water and underwater videos. That no longer matches the intended technical route for this project, which is to use one already-stitched composite video as the only model input for annotation, training, and evaluation.

## What Changes

- **BREAKING** Replace the canonical input contract from synchronized dual-view streams to a single stitched video clip.
- Clarify that `stitched_path` is the required source for baseline manifests, frame extraction, annotation, training, inference, and evaluation.
- Reframe raw above-water and underwater files as optional provenance assets instead of required model inputs.
- Remove baseline assumptions about cross-view synchronization, frame-pair processing, and dual-branch modeling from the technical route.

## Capabilities

### New Capabilities
- None.

### Modified Capabilities
- `breaststroke-keypoint-localization`: Revise the source-data, processing, and evaluation requirements so the localization baseline is defined around one stitched video input instead of synchronized dual-view streams.

## Impact

- Updates the governing OpenSpec requirements for [`openspec/specs/breaststroke-keypoint-localization/spec.md`](/Users/tuian/Documents/大学/竞赛/大创/游泳/code/openspec/specs/breaststroke-keypoint-localization/spec.md).
- Affects manifest, frame extraction, annotation, training, inference, and reporting documentation that still implies dual-view inputs.
- Simplifies the baseline architecture and keeps the project aligned with the existing stitched-video-first dataset and tooling.
