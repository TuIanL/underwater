## Why

The current training pipeline is built around frame-level keypoint localization and requires manually prepared frame indexes and annotations. Phase 1 supervised contrastive pretraining needs a separate video-first workflow so engineers can drop raw swimming videos into a fixed folder layout and automatically derive training-ready clips and stroke labels without hand-maintaining manifests.

## What Changes

- Add a video-ingestion convention for raw swim footage so users can place videos under a fixed directory structure and have athlete, session, and stroke metadata resolved automatically.
- Add a Phase 1 supervised contrastive pretraining workflow that reads single-stroke videos, samples temporal clips, generates two augmented views per sample, and trains an encoder plus projection head with SupCon loss.
- Add validation and filtering rules so mixed-stroke or ambiguous videos can be excluded from Phase 1 datasets before training.
- Add configuration and training entrypoints for the new pretraining path without replacing the existing keypoint localization training flow.

## Capabilities

### New Capabilities
- `stroke-video-ingestion`: Define the repository-managed folder and naming conventions for raw swim videos and the rules for deriving athlete, session, and stroke metadata from those assets.
- `supcon-pretraining`: Define the Phase 1 supervised contrastive pretraining workflow, including clip sampling, dual-view augmentation, stroke-label supervision, and pretraining outputs.

### Modified Capabilities

None.

## Impact

- Affected code: [`src/swim_pose/manifest.py`](/Users/tuian/Documents/大学/竞赛/大创/游泳/code/src/swim_pose/manifest.py), [`src/swim_pose/training/dataset.py`](/Users/tuian/Documents/大学/竞赛/大创/游泳/code/src/swim_pose/training/dataset.py), [`src/swim_pose/training/model.py`](/Users/tuian/Documents/大学/竞赛/大创/游泳/code/src/swim_pose/training/model.py), [`src/swim_pose/training/losses.py`](/Users/tuian/Documents/大学/竞赛/大创/游泳/code/src/swim_pose/training/losses.py), [`src/swim_pose/training/supervised.py`](/Users/tuian/Documents/大学/竞赛/大创/游泳/code/src/swim_pose/training/supervised.py), [`src/swim_pose/cli.py`](/Users/tuian/Documents/大学/竞赛/大创/游泳/code/src/swim_pose/cli.py), and new training configs under `configs/`.
- Affected systems: repository-managed raw video storage, manifest generation, training configuration, and training CLI entrypoints.
- Dependencies: existing PyTorch and OpenCV stack remain sufficient for an initial implementation; no new external service is required.
