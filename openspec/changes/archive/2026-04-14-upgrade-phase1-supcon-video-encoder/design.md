## Context

The repository already has a dedicated Phase 1 SupCon workflow, but two parts of the implementation no longer reflect the team's chosen direction.

First, the dataset currently builds `view_1` and `view_2` by calling the clip-sampling path twice. That means each positive pair can come from different temporal indices within the same source video. The current main spec, however, already defines the pair as two augmented views of the same temporal clip.

Second, the current `TemporalResNetEncoder` is not a true video backbone. It encodes each frame independently with a 2D ResNet and averages features across time. This gives a simple baseline, but it does not explicitly model short-range temporal structure such as stroke phase transitions, coordination timing, or stroke-specific rhythm.

There is also a downstream compatibility boundary that should be made explicit before implementation starts. The current frame-level heatmap localization model uses a 2D ResNet-style stem. A true 3D or factorized video encoder can still be valuable for Phase 1, but its checkpoint should be treated as reusable for compatible video-first consumers by default, not as an implicit drop-in weight source for the existing 2D localization backbone.

## Goals / Non-Goals

**Goals:**
- Make Phase 1 positive pairs use one sampled clip with two independent augmentations.
- Upgrade Phase 1 from frame-wise 2D feature averaging to a temporally aware video encoder.
- Preserve the existing high-level SupCon interface so the rest of the training path still reasons in terms of `features` and `projections`.
- Add experiment-grade configuration knobs for realistic training runs without merging SupCon into the localization entrypoints.
- Clarify what kinds of downstream reuse are guaranteed by the saved Phase 1 checkpoint.

**Non-Goals:**
- Rewriting the current frame-level localization pipeline around a 3D backbone in this change.
- Solving transfer from the new video encoder into the current 2D heatmap model.
- Supporting `different_clips` as a first-class official pairing mode for the main Phase 1 path.
- Starting with a pure video Transformer baseline before a solid video-CNN baseline exists.

## Decisions

### 1. Sample one clip and augment it twice

For each training item, the dataset will:

1. sample temporal indices once,
2. load the raw clip once,
3. run the augmentation pipeline twice to produce `view_1` and `view_2`.

This keeps the time content fixed within a positive pair while still allowing the views to differ through crop, color jitter, grayscale conversion, blur, and similar appearance perturbations.

`temporal_jitter` remains valid in this design, but it applies only to the one clip-sampling decision made for the item. It must not cause `view_1` and `view_2` to diverge temporally after that point.

Why:
- It matches the current spec and the team's chosen contrastive semantics.
- It cleanly separates invariance to visual appearance from invariance to motion-phase differences.

Alternatives considered:
- Keeping two independently sampled clips from the same video. Rejected because it changes the positive-pair meaning and makes the main training path less phase-specific.
- Exposing both modes as equally supported configuration options. Rejected because the spec already defines the canonical behavior and the alternative is better treated as an ablation if ever needed.

### 2. Standardize Phase 1 on a true video encoder

The new encoder will be a temporally aware video model rather than a frame-wise 2D encoder followed only by temporal mean pooling. The first implementation should prefer a torchvision-supported video CNN, with `r2plus1d_18` as the primary baseline candidate and `r3d_18` or `mc3_18` as acceptable fallbacks if needed by environment or stability constraints.

The encoder contract stays simple:
- input: `B x C x T x H x W`
- backbone output: spatiotemporal feature map
- pooling: `AdaptiveAvgPool3d(1)`
- final encoder output: flattened `B x D`

Why:
- Swimming dynamics are inherently temporal and phase-sensitive.
- A video CNN gives stronger temporal inductive bias than a 2D encoder plus averaging, while staying closer to a "strong baseline" than a first-pass Transformer.
- The current dependency set already includes torchvision, so this remains operationally simple.

Alternatives considered:
- Retaining 2D ResNet plus temporal mean pooling. Rejected because it underserves the temporal objective the team has now chosen.
- Adding a light temporal head on top of 2D frame features. Deferred because it is less direct than a true video backbone and risks landing in an ambiguous middle ground.
- Starting with a video Transformer. Rejected for the first upgrade because it raises data, tuning, and hardware demands too early.

### 3. Keep the SupCon model interface stable

`SupConPretrainingModel` will continue to return:
- `features`: pooled encoder output shaped `B x D`
- `projections`: projection-head output shaped `B x P`, L2-normalized

This stabilizes the rest of the training path and lets the dataset and loss changes remain conceptually local.

Why:
- The existing training loop already expects this contract.
- The projection head remains pretraining-only and does not need to leak into downstream inference contracts.

### 4. Narrow checkpoint reuse guarantees to compatible video consumers

This change will explicitly define Phase 1 checkpoint reuse in video-first terms. Saving the encoder separately remains valuable, but the guaranteed reuse target is a compatible downstream video model or video-level consumer, not the current 2D heatmap localization architecture.

Why:
- The existing localization backbone is 2D and uses a different parameter structure.
- Leaving this ambiguous would create false expectations about direct transfer from a 3D checkpoint into the current localization path.

Alternatives considered:
- Preserving an implied promise of direct reuse in the current 2D heatmap model. Rejected because the architectures are no longer aligned once the encoder becomes truly spatiotemporal.
- Adding a transfer bridge in the same change. Rejected because that is a separate research and engineering problem.

### 5. Promote the SupCon config from smoke-oriented to experiment-oriented

The dedicated SupCon config will continue to live separately from localization configs, but it should grow enough control surface for realistic experiments. The expected additions include:
- explicit video backbone selection,
- optional mixed precision,
- gradient accumulation,
- optional gradient clipping,
- optional warmup before cosine decay,
- updated default training values better suited to GPU-backed runs.

Why:
- The current default config is still closer to a smoke or local sanity run than to a serious contrastive experiment.
- A true video backbone increases memory pressure and makes stability controls more important.

### 6. Validate semantics, not just code paths

Verification should move beyond "training runs and writes a checkpoint" and include at least one deterministic proof that both positive views were built from the same sampled clip. The implementation may expose this through a helper, a debug field, or a test seam, but the semantic contract should be directly testable.

Why:
- This change exists specifically to tighten semantics, not just to swap one implementation for another.

## Risks / Trade-offs

- [Higher memory and compute cost] -> Mitigation: start with a modest video-CNN backbone, keep clip length configurable, and expect lower spatial resolution or accumulation compared with the 2D baseline.
- [Checkpoint incompatibility with the current 2D localization model] -> Mitigation: document the boundary explicitly and treat any transfer bridge as follow-up work.
- [Same-clip pairing reduces within-item temporal diversity] -> Mitigation: rely on re-sampling across epochs and batch diversity across samples rather than baking phase variation into the positive pair itself.
- [Torchvision video-model differences across environments] -> Mitigation: anchor the first implementation to the project's locked torchvision version and retain a fallback baseline choice.

## Migration Plan

1. Update the `supcon-pretraining` spec delta to reflect same-clip semantics, true video encoding, and explicit reuse boundaries.
2. Change the dataset so each item samples and loads one clip, then augments it twice.
3. Replace the current temporal mean-pooled 2D encoder with the selected video backbone while keeping the projection-head interface stable.
4. Extend the dedicated SupCon config and training loop with experiment-oriented controls.
5. Run semantic tests plus smoke training, then run a small GPU pilot before wider experiments.

## Open Questions

- Should the first default video backbone be `r2plus1d_18` everywhere, or should the project keep `r3d_18` as an easier fallback on lower-resource hardware?
- Does the team want class-aware batch construction in a follow-up change to reduce the chance of weak positive coverage beyond the two augmented views?
- What is the first concrete downstream video-level consumer expected to reuse the Phase 1 encoder checkpoint?
