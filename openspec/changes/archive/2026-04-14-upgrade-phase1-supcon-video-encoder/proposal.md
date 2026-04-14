## Why

The current Phase 1 SupCon path is good enough as a smoke-tested baseline, but it no longer matches the team's intended semantics or temporal modeling goals.

Today, each training item samples `view_1` and `view_2` by independently re-sampling clip indices from the same source video. That weakens the intended "same clip, two augmentations" positive-pair contract and makes it harder to isolate invariance to visual perturbations from invariance to different motion phases.

The current encoder also applies a 2D ResNet frame-by-frame and then averages features across time. That is lightweight and easy to run, but it underserves phase-sensitive swimming dynamics and falls short of the team's decision to make Phase 1 more explicitly temporal.

## What Changes

- Modify the Phase 1 SupCon dataset so each item samples one temporal clip, loads it once, and derives both positive views from that same raw clip via independent augmentation.
- Replace the current 2D-plus-temporal-mean encoder with a true video encoder, with a torchvision-backed video CNN baseline such as `r2plus1d_18` preferred for the first implementation.
- Keep the SupCon model contract stable at the interface level: `B x C x T x H x W` in, pooled `B x D` features out, then an L2-normalized projection head for contrastive loss.
- Expand the dedicated SupCon config and training loop from smoke-oriented defaults toward experiment-oriented runs, including explicit video-backbone selection and training-stability controls.
- Clarify checkpoint reuse semantics so Phase 1 guarantees reusable video-encoder weights for compatible downstream video consumers, without implying direct drop-in reuse inside the current 2D heatmap localization model.

## Capabilities

### Modified Capabilities

- `supcon-pretraining`: Tighten dual-view sampling semantics to same-clip pairing, upgrade the encoder to a true video model, and clarify experiment-grade configuration plus checkpoint reuse boundaries.

## Impact

- Affected code: [`src/swim_pose/training/dataset.py`](/Users/tuian/Documents/大学/竞赛/大创/游泳/code/src/swim_pose/training/dataset.py), [`src/swim_pose/training/model.py`](/Users/tuian/Documents/大学/竞赛/大创/游泳/code/src/swim_pose/training/model.py), [`src/swim_pose/training/supcon.py`](/Users/tuian/Documents/大学/竞赛/大创/游泳/code/src/swim_pose/training/supcon.py), [`src/swim_pose/training/config.py`](/Users/tuian/Documents/大学/竞赛/大创/游泳/code/src/swim_pose/training/config.py), [`configs/supcon.toml`](/Users/tuian/Documents/大学/竞赛/大创/游泳/code/configs/supcon.toml), and [`tests/test_supcon_pipeline.py`](/Users/tuian/Documents/大学/竞赛/大创/游泳/code/tests/test_supcon_pipeline.py).
- Affected systems: Phase 1 SupCon data sampling, video encoder selection, SupCon checkpoint semantics, and experiment configuration.
- Dependencies: the current PyTorch and torchvision stack remains sufficient for an initial video-CNN upgrade; no external service is required.
