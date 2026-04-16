## Context

Phase 1 SupCon pretraining already uses a true video backbone, and the supervised localization path already contains an explicit bridge mechanism that distills video features into the frame-level heatmap model. That means the repository has already crossed the architectural boundary from pure 2D pretraining to a mixed video-to-2D system.

The remaining weaknesses are not primarily about backbone choice. They are about whether the current pipeline preserves meaningful temporal information once the video checkpoint is reused:
- the SupCon augmentation path still lacks an explicit splash-style temporal occlusion operator,
- the bridge dataset can assemble a "clip" from sparse annotation rows rather than a dense frame stream,
- and the current configs do not make stage-by-stage adoption or rollback obvious.

The dataset layout in this repository makes that second issue especially important. `annotation_index.csv` is sparse and suitable for supervision targets, while `unlabeled_index.csv` is dense and suitable for temporal context. Reusing sparse labeled rows as a proxy for dense clip context would preserve the bridge interface but degrade its semantics.

## Goals / Non-Goals

**Goals:**
- Preserve the current frame-level localization inference and pseudo-label contract.
- Improve Phase 1 robustness to swimmer splash and short-lived water occlusion with a configurable tube-masking augmentation.
- Ensure video-to-2D transfer uses real dense temporal context around labeled center frames.
- Introduce staged rollout controls so each conservative upgrade can be enabled, evaluated, and rolled back independently.
- Keep regression attribution simple by promoting the upgrade one stage at a time.

**Non-Goals:**
- Replacing the downstream 2D heatmap model with a 3D video heatmap predictor.
- Changing exported prediction rows from one target frame to a multi-frame output format.
- Treating a Phase 1 video checkpoint as directly loadable into the 2D localization model.
- Expanding this change into a broad semi-supervised redesign beyond compatibility safeguards.

## Decisions

### 1. Keep downstream localization frame-centric and use video only as a conservative teacher path

The conservative upgrade will continue to treat the video encoder as a pretraining or bridge-time teacher while the deployed localization model remains a frame-level predictor.

This preserves the current inference, evaluation, and pseudo-label interfaces and avoids forcing simultaneous changes across:
- model construction,
- checkpoint loading semantics,
- prediction export,
- pseudo-label generation,
- and downstream visualization tools.

Alternatives considered:
- Replace the 2D localization model with a 3D center-frame predictor now.
  Rejected because it would expand the change from a conservative upgrade into a downstream architecture rewrite.
- Continue allowing implicit partial checkpoint reuse.
  Rejected because the repository already formalized explicit bridge semantics instead of direct compatibility.

### 2. Stage 1 adds configurable tube masking only to Phase 1 SupCon augmentation

Tube masking will be added inside the existing video augmentation pipeline as a configurable operator that masks one spatial region across a short contiguous frame span. It will be treated as a water-occlusion stressor, not as a universal augmentation that must fire on every sample.

The operator will be controlled by explicit config fields such as probability, temporal span, spatial size, fill mode, and optional center bias. The implementation will allow a no-op configuration so the team can revert to the prior augmentation behavior without code changes.

Alternatives considered:
- Per-frame random erasing.
  Rejected because it does not match the temporal coherence of splash occlusion.
- Hard-enable tube masking in the default code path with no off switch.
  Rejected because it would make attribution and rollback harder.

### 3. Stage 2 rebuilds bridge clips from dense frame context around a labeled center frame

When bridge training is enabled, the clip fed to the video teacher must come from a dense temporal source keyed by the labeled frame's `clip_id`, `source_view`, and `frame_index`. The preferred source is a dense extracted-frame index that already maps those identifiers to image paths. If dense extracted frames are unavailable for a sample, the bridge path must degrade safely by skipping the bridge loss or disabling bridge supervision for that sample rather than synthesizing a clip by repeating the labeled frame.

This keeps the bridge semantically honest: the teacher sees true motion context, while the student still predicts only the labeled center frame.

Alternatives considered:
- Keep using sparse annotation rows to approximate clip context.
  Rejected because it silently collapses temporal supervision into duplicate or near-duplicate frames.
- Decode bridge clips directly from raw video for every sample.
  Not preferred as the primary path because the repository already uses extracted-frame indexes for localization workflows, but it remains a fallback option if dense frame manifests are unavailable in a future environment.

### 4. Stage 3 adds rollout governance instead of changing the whole training stack at once

The final stage will expose conservative-upgrade feature toggles and experiment metadata so the team can promote the change in controlled steps:
- Stage 1 can be enabled without touching localization training.
- Stage 2 can be enabled without changing localization inference outputs.
- Stage 3 documents recommended presets, default-safe values, and rollback switches.

This stage also formalizes adoption criteria:
- promote Stage 1 only if SupCon training remains stable and downstream bridge experiments do not regress;
- promote Stage 2 only if bridge-enabled supervised runs outperform or match the prior baseline on held-out metrics;
- keep Stage 3 defaults safe enough that disabling new flags returns the previous stable path.

Alternatives considered:
- Ship all changes behind one global "new pipeline" switch.
  Rejected because it hides which sub-change caused a regression.

## Risks / Trade-offs

- [Tube masking may over-regularize small clips and weaken contrastive training] -> Mitigation: start with low probability and short spans, log enabled settings in checkpoints, and preserve a config-only rollback to the previous augmentation policy.
- [Dense temporal context may be missing or mismatched for some labeled frames] -> Mitigation: require bridge context lookup by `clip_id`, `source_view`, and `frame_index`, and disable bridge supervision per sample when a valid dense clip cannot be assembled.
- [The bridge path becomes more operationally complex because it depends on two dataset granularities] -> Mitigation: document sparse-vs-dense roles explicitly and keep the non-bridge supervised path unchanged.
- [Staged flags increase config surface area] -> Mitigation: group new fields under existing SupCon and bridge sections, ship documented presets, and keep defaults conservative.
- [Stage interactions may still obscure which change improved results] -> Mitigation: checkpoint each stage separately and compare against the previous stable stage rather than only against the original baseline.

## Migration Plan

### Stage 1: Tube masking in SupCon pretraining

Benefit:
- Improves robustness to short-lived splash occlusion without changing downstream model interfaces.

Primary risk:
- Excessive masking can slow convergence or push the encoder toward shortcut-resistant but less discriminative features.

Rollback point:
- Disable tube masking through config and continue using the prior crop/color/blur augmentation policy.

Adoption notes:
- Roll out first in experiment configs, not as an irreversible behavioral default.
- Save checkpoints with augmentation metadata so Stage 1 and pre-Stage-1 runs remain distinguishable.

### Stage 2: Dense-frame bridge clip sourcing

Benefit:
- Makes bridge distillation use genuine temporal evidence around labeled center frames, which is the main reason to keep the video teacher in the loop.

Primary risk:
- Missing or inconsistent dense frame indexes can reduce bridge coverage or silently produce wrong temporal neighbors if lookup rules are loose.

Rollback point:
- Turn off bridge training and fall back to the existing frame-only supervised localization path while retaining any accepted Stage 1 improvements.

Adoption notes:
- Do not fall back to repeated center-frame clips when dense context is missing.
- Treat "bridge unavailable for this sample" as an explicit condition, not as an implicit silent approximation.

### Stage 3: Config, experiment gating, and rollback discipline

Benefit:
- Makes the conservative upgrade reproducible, easier to bisect, and safer to adopt in shared experiments.

Primary risk:
- More toggles can confuse users if names and defaults are unclear.

Rollback point:
- Revert to the documented baseline config preset with all conservative-upgrade toggles disabled.

Adoption notes:
- Stage 3 is complete only when a teammate can read the config and tell which stage is enabled without reading code.

## Open Questions

- Should dense bridge context be sourced exclusively from a dedicated dense frame manifest, or should the implementation also support on-demand reconstruction from the extracted frame tree when that manifest is absent?
- How aggressive should the initial tube-masking default be for 8-frame clips: low-probability black boxes, noise fill, or a small preset sweep?
