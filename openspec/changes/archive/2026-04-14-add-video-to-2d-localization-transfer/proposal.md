## Why

The upgraded Phase 1 SupCon path now saves a true video-encoder checkpoint and explicitly stops promising direct drop-in reuse inside the current 2D heatmap localization model.

If the team still wants to benefit from Phase 1 pretraining in the existing localization pipeline, that transfer path needs to be designed intentionally rather than relying on partial `strict=False` weight loading.

## What Changes

- Define a supported bridge from Phase 1 video pretraining outputs into the current frame-level localization pipeline.
- Decide whether the bridge should use frame-level distillation, teacher-student supervision, partial weight projection, or another explicit transfer mechanism.
- Add clear checkpoint and configuration semantics so localization training can opt into the bridge without confusing video checkpoints for directly compatible 2D backbones.

## Capabilities

### Modified Capabilities

- `breaststroke-keypoint-localization`: add an explicit, optional transfer path from Phase 1 video pretraining into the 2D localization model instead of relying on implicit checkpoint compatibility.
