# Localization Bridge Spike

This spike compares the current supervised localization baseline against the explicit video-to-2D bridge path.

## Goal

Test whether a frozen Phase 1 SupCon video teacher improves held-out localization quality when its clip-level features distill into the 2D heatmap student during supervised training.

## Configs

- Baseline: `configs/supervised.toml`
- Bridge spike: `configs/supervised_bridge.toml`

## Commands

Train the baseline:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run -- swim-pose train supervised --config configs/supervised.toml
```

Run baseline inference on the validation split:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run -- swim-pose predict \
  --config configs/supervised.toml \
  --checkpoint artifacts/checkpoints/supervised/best.pt \
  --index data/manifests/splits/val.csv \
  --output artifacts/reports/supervised_val_predictions.jsonl
```

Evaluate the baseline:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run -- swim-pose evaluate \
  --predictions artifacts/reports/supervised_val_predictions.jsonl \
  --annotations data/manifests/annotation_index.csv \
  --output artifacts/reports/supervised_val_report.json
```

Train the bridge model:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run -- swim-pose train supervised --config configs/supervised_bridge.toml
```

Run bridge inference on the validation split:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run -- swim-pose predict \
  --config configs/supervised_bridge.toml \
  --checkpoint artifacts/checkpoints/supervised_bridge/best.pt \
  --index data/manifests/splits/val.csv \
  --output artifacts/reports/supervised_bridge_val_predictions.jsonl
```

Evaluate the bridge model:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run -- swim-pose evaluate \
  --predictions artifacts/reports/supervised_bridge_val_predictions.jsonl \
  --annotations data/manifests/annotation_index.csv \
  --output artifacts/reports/supervised_bridge_val_report.json
```

## Baseline Comparison

Use the same train/val split, seed, and student backbone for both runs. The only intentional difference is the bridge path enabled by `configs/supervised_bridge.toml`.

## Success Criteria

Treat the spike as promising if all of the following hold:

- Validation `mean_normalized_error` improves by at least 5% relative to the baseline, or `pck@0.05` improves by at least 0.01 absolute.
- The bridge run does not regress `visible_mean_error` on difficult joints such as wrists, ankles, heels, and toes when spot-checking the evaluation output.
- Training remains numerically stable and produces a `localization_bridge` checkpoint without confusing it for a direct video-pretraining artifact.

If the bridge fails those checks, keep the change open only as a research path and avoid promoting it into the default localization workflow.
