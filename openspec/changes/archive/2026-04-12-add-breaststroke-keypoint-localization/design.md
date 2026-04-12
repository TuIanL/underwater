## Context

The project's first usable milestone is not stroke scoring or coaching feedback, but reliable joint localization for professionally captured breaststroke videos. The available data is unusually strong for this task: side-view breaststroke recordings captured by synchronized above-water and underwater cameras, with full-stroke videos rather than isolated images. The main constraints are limited labeling bandwidth, waterline transitions, bubbles, reflections, and frequent self-occlusion during the stroke cycle.

The design therefore focuses on a narrow but foundational outcome: a dual-view 2D keypoint localization baseline that produces anatomically meaningful landmarks, visibility states, and confidence values that later stages can trust.

## Goals / Non-Goals

**Goals:**
- Define a canonical 18-point keypoint schema tailored to breaststroke analysis.
- Define annotation semantics that remain consistent under occlusion, waterline transitions, and side-view ambiguity.
- Define a data usage strategy that prioritizes synchronized raw views over stitched composites.
- Define a semi-supervised training approach that combines pretrained pose features, a small high-quality labeled seed set, and large unlabeled breaststroke videos.
- Define evaluation criteria that measure per-joint accuracy, occluded-joint behavior, temporal stability, and cross-athlete generalization.

**Non-Goals:**
- Stroke error detection, coaching feedback, or quality scoring.
- 3D pose reconstruction or camera calibration for metric reconstruction.
- Multi-person tracking, general swim-lane analytics, or support for all four strokes.
- Dependence on stitched video as the only model input when raw synchronized views are available.
- Commitment to a single final model family before baseline experiments are run.

## Decisions

### 1. Use synchronized raw above-water and underwater streams as canonical inputs

The project SHALL treat the two original synchronized camera streams as the source of truth for training and evaluation. The stitched video remains useful for playback, demo material, and manual review, but it introduces a visible seam at the waterline, geometric mismatch, and domain discontinuity that can confuse keypoint learning.

Alternatives considered:
- Train directly on stitched videos only: simpler data handling, but higher risk of seam artifacts and unstable waterline landmarks.
- Use only the underwater stream: easier lower-body visibility, but weaker coverage for head and water-surface interactions.

### 2. Keep phase one strictly at 2D keypoint localization

The first change only targets frame-level 2D keypoints, point visibility, and confidence. This keeps the milestone aligned with the hardest current problem and avoids mixing supervision for scoring or diagnosis before reliable landmarks exist.

Alternatives considered:
- Start with end-to-end stroke quality scoring: rejected because it hides failure modes and needs harder labels.
- Start with action-phase classification: useful later, but secondary to accurate landmark geometry.

### 3. Standardize on an 18-point breaststroke schema

The initial annotation contract uses the following landmarks:

| Group | Keypoints | Definition |
| --- | --- | --- |
| Head | `nose`, `neck` | `nose` is the nose tip or closest visible nasal profile point; `neck` is the midpoint at the base of the neck between the shoulders |
| Upper limbs | `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist` | Centers of the corresponding anatomical joints |
| Lower limbs | `left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_ankle`, `right_ankle` | Centers of the corresponding anatomical joints |
| Feet | `left_heel`, `right_heel`, `left_toe`, `right_toe` | Posterior heel center and most distal visible toe tip or forefoot tip |

Left and right SHALL always refer to the athlete's anatomical left and right, never image left and right. This rule remains in effect even when one side is farther from the camera, temporarily hidden, or visually overlaps with the opposite limb.

Alternatives considered:
- A standard COCO-style 17-point schema: good for reuse, but it omits heel/toe detail that is important for breaststroke kick analysis.
- Adding facial landmarks such as chin or ear immediately: useful for future breathing analysis, but not required for the first localization baseline.

### 4. Use explicit visibility states instead of forcing all points to be "visible"

Each keypoint receives one of three visibility states:

- `2`: visible and directly localizable
- `1`: not directly visible but inferable from anatomy and nearby frames
- `0`: not visible and not reliably inferable because of full occlusion, severe splash, or leaving the frame

This prevents the training set from encoding false certainty in difficult frames and creates a clean contract for semi-supervised confidence filtering.

Alternatives considered:
- Binary visible/not visible labels: simpler, but loses the distinction between "occluded but inferable" and "unknown."
- Always annotate every keypoint regardless of confidence: rejected because it increases label noise in precisely the hardest regions.

### 5. Sample annotations by stroke cycle and difficulty, not by dense adjacent frames

The initial manual labeling set should prioritize diversity over raw frame count. The seed set should cover all major phases of the breaststroke cycle, multiple athletes, and difficult examples near the waterline, with bubbles, and with strong limb overlap. A practical target is roughly 1,000 to 2,000 corrected keyframes before large-scale pseudo-labeling begins.

Dataset partitions SHALL be created by athlete and capture session, never by random frame-level splits, to avoid inflated validation scores from near-duplicate frames.

Alternatives considered:
- Uniform frame sampling through each video: easy to automate, but produces many redundant frames.
- Random frame splits across the whole corpus: invalid for generalization measurement.

### 6. Use pretrained pose features plus semi-supervised learning as the main training path

The baseline training path starts from a general human pose pretrained model rather than random initialization. The supervised seed set establishes the anatomical contract, then larger unlabeled breaststroke videos are added through pseudo-labeling and consistency constraints.

Recommended training stages:
1. Train a supervised baseline on the seed annotations.
2. Generate pseudo-labels on unlabeled videos using confidence thresholds.
3. Retrain with labeled and pseudo-labeled samples plus augmentation consistency.
4. Add temporal smoothing or temporal consistency loss only after the single-frame baseline is stable.

Recommended self-/semi-supervised signals:
- weak/strong augmentation consistency
- confidence-gated pseudo-labeling
- bone-length consistency across adjacent frames
- temporal smoothness across short sequences
- optional cross-view consistency for synchronized frames

Alternatives considered:
- Purely self-supervised keypoint discovery from scratch: high research risk and weak semantic guarantees.
- Detector-first optimization as the main problem: unnecessary for fixed-camera, single-athlete clips where keypoint precision is the real bottleneck.

### 7. Treat detection and tracking as support utilities, not the core research contribution

If the crop around the athlete is stable, fixed or lightly tracked ROIs are acceptable for the first phase. A detector such as YOLO may be used as a preprocessing utility, but the design does not make detector choice the primary axis of innovation.

Alternatives considered:
- A detector-centric pipeline as the headline approach: valuable for deployment later, but not where current technical risk lies.

### 8. Evaluate not only static accuracy, but also stability under occlusion and across athletes

The project will report:

- normalized keypoint localization error
- PCK-style accuracy at one or more thresholds
- per-joint accuracy, especially wrists, ankles, heels, and toes
- visible vs occluded point performance
- temporal jitter or frame-to-frame instability
- held-out athlete and held-out session results

This prevents misleading progress claims based on easy frames or within-athlete leakage.

Alternatives considered:
- Reporting only aggregate validation loss: not interpretable enough for later clinical or coaching use.

## Risks / Trade-offs

- [Raw streams may not be perfectly synchronized] -> Maintain a clip manifest with synchronization notes and exclude badly misaligned pairs from cross-view experiments.
- [Heel/toe labels may be ambiguous in splash-heavy frames] -> Provide example-driven annotation guidance and use visibility state `0` or `1` instead of forced precise coordinates.
- [Professional athletes may reduce appearance diversity] -> Report generalization limits explicitly and expand to broader body types in later changes.
- [Pseudo-labels can reinforce early model mistakes] -> Use confidence thresholds, periodic manual audits, and active correction of low-confidence or high-disagreement samples.
- [Dual-view data handling increases pipeline complexity] -> Keep a fallback single-view baseline so the project can progress even if synchronization tooling lags.

## Migration Plan

1. Create dataset manifests linking synchronized raw clips, stitched review clips, athlete IDs, and session IDs.
2. Publish the annotation guide and train annotators on the 18-point schema and visibility semantics.
3. Produce the first seed annotation set and establish review checks.
4. Train and benchmark a supervised localization baseline.
5. Add semi-supervised expansion and compare against the supervised baseline on held-out athletes.

## Open Questions

- Are both original raw camera streams available and time-aligned for all collected videos?
- Will the first baseline train one shared model across both views, or separate per-view models with late comparison?
- Is an additional breathing-related landmark such as `chin` worth introducing in a future change once localization is stable?
