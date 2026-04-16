## Conservative Upgrade Notes

### Rollout Presets

- Stage 0 SupCon rollback preset: [configs/supcon.stage0.toml](/Users/tuian/Documents/大学/竞赛/大创/游泳/code/configs/supcon.stage0.toml)
- Stage 1 SupCon preset with tube masking: [configs/supcon.toml](/Users/tuian/Documents/大学/竞赛/大创/游泳/code/configs/supcon.toml)
- Stage 0 frame-only localization preset: [configs/supervised.toml](/Users/tuian/Documents/大学/竞赛/大创/游泳/code/configs/supervised.toml)
- Stage 2 dense-context bridge preset: [configs/supervised_bridge.toml](/Users/tuian/Documents/大学/竞赛/大创/游泳/code/configs/supervised_bridge.toml)

### Rollback Switches

- Stage 1 rollback: use [configs/supcon.stage0.toml](/Users/tuian/Documents/大学/竞赛/大创/游泳/code/configs/supcon.stage0.toml) or set `tube_mask_prob = 0.0`
- Stage 2 rollback: use [configs/supervised.toml](/Users/tuian/Documents/大学/竞赛/大创/游泳/code/configs/supervised.toml) or set `bridge.enabled = false`
- Missing dense context policy: keep `skip_missing_context = true` to skip bridge loss explicitly instead of fabricating clips

## Validation Notes

### Stage 0 vs Stage 1 SupCon CPU smoke

Environment:
- Actual `data/manifests/supcon_videos.csv`
- CPU-only smoke run
- `64x64`, `clip_length = 4`, `batch_size = 2`, `epochs = 1`
- `pretrained_backbone = false` to avoid download-dependent variance

Results:
- Stage 0 baseline: loss `0.9770`, elapsed `30.918s`
- Stage 1 tube masking: loss `1.0632`, elapsed `38.477s`

Observed benefit:
- Stage 1 completed successfully with tube masking enabled and wrote the expected stage metadata into the checkpoint, so the conservative augmentation path is stable enough for opt-in experimentation.

Main risk:
- Tube masking increased runtime in this CPU smoke and modestly increased contrastive loss, so it should remain configurable rather than treated as an unconditional default for every environment.

Rollback point:
- Revert to [configs/supcon.stage0.toml](/Users/tuian/Documents/大学/竞赛/大创/游泳/code/configs/supcon.stage0.toml)

### Stage 1 vs Stage 1+2 localization smoke

Environment:
- `data/manifests/annotation_index.csv` currently has 6 rows
- Repository `val.csv` and `test.csv` are empty, so a temporary `4 train / 2 val` frame-held-out split was created for this smoke check
- CPU-only smoke run
- Student image size `64x64`, heatmaps `16x16`, `epochs = 5`, `batch_size = 2`
- Dense bridge context sourced from `data/manifests/unlabeled_index.csv`

Results:
- Stage 1 frame-only: mean normalized error `0.1404`, `PCK@0.10 = 0.2759`, elapsed `1.177s`
- Stage 1+2 dense bridge: mean normalized error `0.1258`, `PCK@0.10 = 0.5517`, elapsed `7.211s`

Observed benefit:
- On the temporary held-out split, dense bridge context improved mean normalized error and materially improved `PCK@0.10`, which is enough evidence to keep Stage 2 available as an opt-in path.

Main risk:
- The comparison is only a weak held-out smoke because the repository does not yet have a populated validation or test split. Adoption should therefore stay conservative until a stronger held-out split exists.

Rollback point:
- Revert to [configs/supervised.toml](/Users/tuian/Documents/大学/竞赛/大创/游泳/code/configs/supervised.toml)

## Adoption Guidance

- Stage 1 is ready for opt-in use because it is config-gated and survived the SupCon smoke run.
- Stage 2 is promising but should remain non-default until the team has a real held-out split with rows in `val.csv` or `test.csv`.
- Stage 3 is complete only insofar as rollout and rollback are now config-driven; stronger acceptance still depends on better evaluation data.
