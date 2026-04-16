## Why

The repository already moved Phase 1 SupCon pretraining onto a true video backbone, but the surrounding transfer path is still fragile in ways that can erase that temporal benefit. The current bridge path can assemble clip context from sparsely labeled rows instead of dense frame streams, and the video augmentation policy still lacks an explicit water-occlusion stressor such as tube masking.

This change proposes a conservative upgrade rather than a full downstream rewrite. The goal is to preserve the current 2D localization inference contract while making the Phase 1 video signal more trustworthy, more stress-tested against splash occlusion, and easier to disable or roll back if experiments regress.

## What Changes

- Strengthen Phase 1 SupCon augmentation with an explicit water-aware temporal occlusion operator, starting with configurable tube masking applied to short contiguous frame spans.
- Tighten the bridge path from video pretraining into localization so clip-conditioned supervision is sourced from dense frame streams rather than only from sparse annotation rows.
- Keep the downstream localization model, inference output shape, and pseudo-label contract frame-centric; this change does not introduce a 3D heatmap predictor.
- Add staged rollout guidance so the team can adopt the upgrade incrementally:
  - Stage 1: add tube masking and augmentation controls to SupCon pretraining.
  - Stage 2: correct bridge clip assembly to use dense temporal context around labeled center frames.
  - Stage 3: add training/config safeguards, experiment toggles, and explicit rollback paths for each conservative upgrade.
- Clarify experiment expectations for each stage, including expected upside, principal regression risk, and the condition for falling back to the prior baseline.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `supcon-pretraining`: strengthen water-aware augmentation semantics and require staged, configurable training-stability controls for conservative temporal upgrades.
- `breaststroke-keypoint-localization`: require explicit dense-frame temporal context when the localization pipeline opts into video-to-2D transfer, while preserving the existing frame-level prediction contract.

## Impact

- Affected code: `src/swim_pose/training/dataset.py`, `src/swim_pose/training/supcon.py`, `src/swim_pose/training/supervised.py`, `src/swim_pose/training/bridge.py`, `src/swim_pose/training/inference.py`, `src/swim_pose/training/pseudolabels.py`, `configs/supcon.toml`, and `configs/supervised_bridge.toml`.
- Affected systems: Phase 1 SupCon augmentation, video-to-2D transfer during supervised localization, training configuration, and experiment rollback discipline.
- Dependencies: existing PyTorch and torchvision video support remains sufficient; no external services are required.
