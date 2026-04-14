## 1. Dataset Semantics

- [x] 1.1 Update `SupConVideoDataset` so each item samples temporal indices once, loads one raw clip, and derives `view_1` and `view_2` from that same clip
- [x] 1.2 Ensure temporal jitter applies only to the one clip-sampling step and keep augmentation randomness independent per view
- [x] 1.3 Add verification that proves both views share the same temporal source clip while remaining visually non-identical after augmentation

## 2. Video Encoder Upgrade

- [x] 2.1 Replace the current temporal mean-pooled 2D encoder with a true video encoder that accepts `B x C x T x H x W` and returns pooled `B x D` features
- [x] 2.2 Keep `SupConPretrainingModel` and the projection-head contract stable so the training loop still consumes `features` and normalized `projections`
- [x] 2.3 Save checkpoints in a way that preserves reusable video-encoder weights and clearly excludes any implicit promise of direct reuse inside the current 2D localization model

## 3. Experiment Configuration And Training

- [x] 3.1 Extend the dedicated SupCon config with explicit video-backbone selection and experiment-stability controls such as accumulation, AMP, clipping, and warmup as implemented
- [x] 3.2 Update the SupCon training loop to honor the new settings without changing the existing localization entrypoints
- [x] 3.3 Refresh the default SupCon config so it reads as a realistic experiment baseline rather than a CPU-oriented smoke configuration

## 4. Validation

- [x] 4.1 Expand tests beyond ingestion and checkpoint smoke coverage to include same-clip pairing semantics and video-encoder forward coverage
- [x] 4.2 Run a small GPU-backed pilot to confirm memory profile, loss stability, and checkpoint writing with the selected video backbone
- [x] 4.3 Capture any future work needed for transfer into the 2D localization pipeline as a separate follow-up change, if the team still wants that path
