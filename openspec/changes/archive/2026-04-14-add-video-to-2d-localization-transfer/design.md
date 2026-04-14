## Context

Phase 1 now uses a true video encoder for supervised contrastive pretraining. That is the right choice for temporal representation learning, but it creates a clean architectural split from the current frame-level heatmap localization model, which still uses a 2D ResNet-style stem.

The old habit of pointing `pretrained_checkpoint` at a SupCon checkpoint and hoping partial key matches transfer something useful is no longer a supported strategy. If the team wants Phase 1 to improve localization, it needs a deliberate bridge.

## Goals / Non-Goals

**Goals:**
- Make any transfer from Phase 1 video pretraining into localization explicit and reproducible.
- Preserve the current localization inference contract while allowing better initialization or auxiliary supervision.
- Prevent silent misuse of incompatible checkpoints.

**Non-Goals:**
- Replacing the 2D localization backbone with a 3D backbone as part of the bridge.
- Reworking the current Phase 1 video encoder design.

## Decisions

### 1. Treat transfer as a teacher-student or representation-bridge problem

The bridge should be implemented as an explicit training mechanism, not as direct checkpoint loading. Reasonable options include:
- frame-level distillation from video-encoder features into a 2D student,
- clip-conditioned auxiliary supervision during localization training,
- a learned projection from video-pretrained representations into 2D feature space.

### 2. Keep localization checkpoints and video-pretraining checkpoints distinct

The localization pipeline should continue to consume localization-native checkpoints. Any bridge artifacts should have their own metadata so users can tell whether they are loading:
- a video-pretraining checkpoint,
- a bridge checkpoint,
- or a localization checkpoint.

### 3. Stage this as follow-up research work

The first step should be a small design-and-spike effort that compares at least two transfer strategies before implementation starts.

## Risks / Trade-offs

- [Transfer may add training complexity without clear gain] -> Mitigation: spike on a small dataset before committing to a full bridge.
- [Users may confuse bridge artifacts with localization checkpoints] -> Mitigation: keep metadata and config names explicit.

## Open Questions

- Which bridge objective is the most sample-efficient for the current dataset size?
- Should the bridge happen during supervised localization only, or also during semi-supervised training?
