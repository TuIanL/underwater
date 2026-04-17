## Context

The repository's current localization path is built around a custom frame-level heatmap model with optional semi-supervised training and an optional video-to-2D bridge from Phase 1 SupCon pretraining. That stack already preserves a frame-keyed inference contract, but it is expensive to extend and does not directly attack the two failure modes showing up most often in experiments: underwater occlusion and frame-to-frame flicker on stitched footage.

At the same time, the project still depends on a swim-specific 18-point schema, visibility semantics, and downstream biomechanics consumers that expect interpretable trajectories rather than only raw model logits. The architecture shift therefore needs to improve the baseline model family without breaking the repository's current data contracts or abandoning the existing research work outright.

## Goals / Non-Goals

**Goals:**
- Make a pretrained YOLOv8-Pose-style 2D model the default localization baseline for stitched swim frames.
- Preserve the project's 18-point anatomical schema, frame-keyed exports, confidence values, and visibility-related semantics.
- Focus adaptation work on underwater and stitched-video failure modes such as splash, blur, seam artifacts, and self-occlusion.
- Add a confidence-aware temporal postprocessing stage that produces stabilized trajectories for downstream biomechanics analysis while keeping raw predictions available.
- Keep the current custom heatmap and video-bridge paths runnable as explicit research baselines rather than deleting them.
- Update evaluation so raw and filtered predictions can be compared honestly on the same held-out splits.

**Non-Goals:**
- Training a 3D or end-to-end temporal pose model from scratch as part of this change.
- Changing the canonical phase-one input away from stitched videos.
- Replacing the 18-point project schema with an external dataset's smaller keypoint definition.
- Converting every downstream tool to consume multi-frame prediction volumes.
- Delivering final stroke scoring or coaching logic in this change.

## Decisions

### 1. Make the pretrained YOLO pose stack the default supervised baseline, but keep it behind project-owned CLI and config surfaces

The project will adopt a YOLOv8-Pose-style training and inference path as the new default baseline because it offers strong human-anatomy priors and faster iteration than continuing to scale the current custom heatmap model as the only mainline path. The repository will still expose this through project-owned commands, config files, and export adapters so the rest of the system does not depend directly on vendor-specific artifact shapes.

Alternatives considered:
- Keep the custom ResNet heatmap model as the only default baseline.
  Rejected because it leaves the project carrying most of the modeling burden itself.
- Jump directly to a 3D or temporal model rewrite.
  Rejected because it increases risk before the 2D underwater baseline is stable.

### 2. Preserve the 18-point swim contract through an explicit schema-adaptation layer instead of shrinking to the external pretrained schema

The external pose model may arrive with a pretrained keypoint definition that does not exactly match the project's 18-point schema. This change will therefore treat schema preservation as a first-class contract: dataset conversion, training labels, inference decoding, and exported predictions must still produce the project's 18 keypoints with anatomical left/right semantics. Overlapping upstream joints can reuse pretrained initialization, while project-specific landmarks such as `neck`, `heel`, and `toe` require explicit adaptation rather than silent omission.

Alternatives considered:
- Shrink the project to the external pretrained schema.
  Rejected because it would weaken the downstream biomechanics use case.
- Infer project-only landmarks purely with post-hoc heuristics.
  Rejected because it hides uncertainty and weakens supervision.

### 3. Treat underwater domain adaptation as a data-and-training concern, not as a separate model family

The main adaptation work will live in dataset conversion, augmentation, sampling, and fine-tuning policy. The baseline will prioritize hard underwater cases such as bubbles, splash, motion blur, seam crossings, and partial occlusion, with experiment settings that can be promoted gradually. This keeps the architecture simple while still letting the team focus on the domain shift that matters most.

Alternatives considered:
- Depend on general-purpose human pose pretraining alone.
  Rejected because current failure modes show a meaningful underwater domain gap.
- Split the project into separate above-water and underwater model families now.
  Rejected because it expands scope before the stitched baseline is reliable.

### 4. Add temporal postprocessing as a non-destructive downstream stage

Temporal filtering will be introduced after raw frame-level pose inference rather than replacing it. The pipeline will keep raw predictions as the canonical model output, then derive filtered trajectories using confidence-aware smoothing or tracking logic that can be turned on or off. This lets downstream biomechanics analysis consume stabilized motion while preserving traceability back to the raw detector output.

Alternatives considered:
- Overwrite raw predictions with filtered values.
  Rejected because it hides model behavior and makes debugging harder.
- Handle all stability only inside training.
  Rejected because the repository currently lacks a robust deployment-time stabilization layer and the user need is explicitly trajectory quality.

### 5. Replace naive motion-as-jitter reporting with raw-versus-filtered stability evaluation

The existing frame-to-frame displacement reporting is useful for spotting unstable transitions, but it confounds real swimmer motion with prediction flicker. The new evaluation path will therefore compare raw and filtered predictions on the same held-out data and add stability metrics that operate on residuals, consistency, or physically plausible motion patterns rather than only on absolute displacement.

Alternatives considered:
- Keep the current temporal-jitter metric as the main stability signal.
  Rejected because it over-penalizes genuine stroke motion.
- Evaluate only filtered outputs.
  Rejected because it would hide whether the base detector regressed.

### 6. Keep SupCon and bridge paths as optional research baselines with explicit metadata

The repository will not delete the current custom heatmap, SupCon, or bridge work. Instead, this change will demote them from default localization architecture to optional comparison baselines. Experiment metadata, config names, and documentation should make it obvious whether a run used the YOLO baseline, the legacy heatmap baseline, or an auxiliary bridge experiment.

Alternatives considered:
- Remove the older paths immediately.
  Rejected because they still have research value and provide rollback options.
- Continue presenting all paths as equally primary.
  Rejected because it dilutes the new baseline decision and slows adoption.

## Risks / Trade-offs

- [External YOLO dependency introduces versioning and license review overhead] -> Mitigation: pin the dependency version, document the integration boundary, and review usage constraints before broad distribution.
- [Adapting a pretrained pose stack to the project's 18-point schema may take more work than a drop-in fine-tune] -> Mitigation: stage the work so overlapping keypoints inherit pretrained weights first, while project-specific landmarks are added and validated explicitly.
- [Temporal filtering can oversmooth fast kick dynamics or hide real uncertainty] -> Mitigation: export both raw and filtered trajectories, make filtering configurable, and compare them on held-out sequences before promotion.
- [Underwater domain shift may still dominate with the current label volume] -> Mitigation: prioritize hard-frame annotation, domain-targeted augmentation, and audit loops before expanding into more architectural complexity.
- [Keeping legacy and new baselines side by side increases config surface area] -> Mitigation: label default versus research presets clearly and require run metadata that identifies the active baseline.

## Migration Plan

1. Introduce project-owned dataset conversion and config scaffolding for the YOLO pose baseline while keeping the current training path intact.
2. Train and evaluate the first supervised YOLO-based stitched-video baseline against the current default on the same labeled split.
3. Add underwater-domain adaptation settings and hard-frame curation, then compare again against the plain pretrained baseline.
4. Add temporal postprocessing and dual raw/filtered export, then extend evaluation to report both paths side by side.
5. Reclassify the current custom heatmap and bridge configs as research baselines in documentation and experiment metadata once the new baseline is accepted.

Rollback strategy:
- Keep the current custom heatmap path runnable until the YOLO baseline matches or exceeds it on agreed metrics.
- Keep temporal postprocessing behind an explicit toggle so the system can revert to raw predictions without retraining.

## Open Questions

- Should the first integration target strict YOLOv8-Pose specifically, or a newer Ultralytics pose family that preserves the same project contract?
- How should visibility-related outputs be represented when the upstream pose stack does not natively expose the same three-state visibility contract as the current pipeline?
- Which temporal filter family gives the best trade-off for breaststroke trajectories: One Euro, Kalman-style filtering, Savitzky-Golay smoothing, or a project-specific hybrid?
- What minimum labeled set is required before underwater-domain adaptation meaningfully improves the pretrained baseline rather than overfitting it?
