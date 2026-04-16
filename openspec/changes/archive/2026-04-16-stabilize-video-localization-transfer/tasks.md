## 1. Stage 1 SupCon Tube Masking

- [x] 1.1 Extend the SupCon dataset and config plumbing with explicit tube-masking controls that default to a rollback-safe no-op when disabled
- [x] 1.2 Implement tube masking inside the video augmentation pipeline so it masks one spatial region across a short contiguous frame span without changing the `B x C x T x H x W` training contract
- [x] 1.3 Add verification for tube-masking semantics and the disabled-path rollback behavior

## 2. Stage 2 Dense-Frame Bridge Context

- [x] 2.1 Add dense temporal context lookup keyed by labeled-frame `clip_id`, `source_view`, and `frame_index`
- [x] 2.2 Update bridge-enabled supervised training so teacher clips come from dense context and the bridge loss is skipped explicitly when valid context is unavailable
- [x] 2.3 Add coverage proving the bridge path no longer fabricates temporal clips by repeating sparse labeled frames

## 3. Stage 3 Rollout And Compatibility Controls

- [x] 3.1 Expose stage-specific conservative-upgrade toggles and metadata in SupCon and bridge configs so each stage can be enabled or disabled independently
- [x] 3.2 Preserve frame-keyed localization inference and pseudo-label compatibility when checkpoints were trained with the conservative bridge path
- [x] 3.3 Document or encode the baseline preset and rollback switches so teammates can revert to the last stable stage without source edits

## 4. Validation And Adoption Gates

- [x] 4.1 Run a Stage 0 vs Stage 1 experiment comparison to confirm SupCon stability and acceptable memory behavior before enabling bridge changes
- [x] 4.2 Run a Stage 1 vs Stage 1+2 comparison on held-out localization metrics to decide whether dense bridge context should be adopted
- [x] 4.3 Record the observed benefit, main regression risk, and rollback point for each stage in the experiment notes or change artifacts before implementation is considered complete
