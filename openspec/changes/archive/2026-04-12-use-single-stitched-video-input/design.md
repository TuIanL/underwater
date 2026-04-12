## Context

The archived localization change established a strong annotation and evaluation contract, but its input assumption is now outdated. The current specification says the canonical source is a synchronized pair of above-water and underwater videos, while the working dataset, manifests, frame extraction defaults, and annotation flow in this repository already center on a single stitched clip.

This mismatch creates confusion in technical writing and implementation planning. The design goal for this change is to make the spec reflect the actual baseline route: one stitched composite video is the only required input, and every downstream stage operates on that stitched timeline.

## Goals / Non-Goals

**Goals:**
- Make one stitched video clip the canonical input unit for the localization baseline.
- Define manifest, frame extraction, annotation, training, inference, and evaluation around a single stitched timeline.
- Preserve optional raw camera paths only as metadata or provenance when they exist.
- Remove baseline dependence on synchronization logic, cross-view pairing, and dual-branch modeling.

**Non-Goals:**
- Introduce a new multi-view fusion baseline.
- Remove the 18-point schema, visibility semantics, or athlete/session split rules.
- Solve seam artifacts introduced by the stitched video beyond documenting them and evaluating their impact.
- Redesign the project around 3D reconstruction, calibration, or stroke scoring.

## Decisions

### 1. Use one stitched clip as the canonical input record

The baseline dataset contract will require one stitched video per clip, referenced by `stitched_path`. If raw above-water or underwater files exist, they can stay in the manifest as optional provenance fields, but they are not required to prepare training or evaluation data and must not be treated as the canonical source.

Alternatives considered:
- Keep dual-view streams as canonical and treat stitched clips as fallback only: rejected because it conflicts with the intended method and the current stitched-first workflow.
- Remove raw camera fields entirely: rejected because keeping provenance is still useful for future experiments and data bookkeeping.

### 2. Normalize all sampling and annotation steps to the stitched timeline

Frame extraction, seed-frame selection, annotation indexing, and split generation will all key off the stitched clip and use `source_view = stitched` as the baseline convention. Frame identifiers will refer only to positions in the stitched video.

Alternatives considered:
- Keep separate above/under frame identifiers and map them onto the stitched clip later: rejected because it preserves complexity the baseline no longer needs.

### 3. Define the training and inference baseline as single-stream processing

The baseline model contract will consume frames or short temporal windows from the stitched video only. Cross-view consistency losses, paired-frame encoders, and synchronization-aware branches are out of scope for this baseline and should not appear as required behavior in the spec.

Alternatives considered:
- Maintain a dual-branch model design as the preferred future path: rejected because it weakens the clarity of the immediate technical route.
- Require temporal models immediately: rejected because the baseline can remain single-frame first, with optional temporal refinement later on stitched clips.

### 4. Keep seam-related limitations explicit in evaluation and documentation

The stitched video introduces a waterline seam and geometric discontinuity, especially around the torso, wrists, and kick trajectory. Rather than hiding this, the design keeps stitched-video evaluation as the primary benchmark and requires reporting that makes seam-driven failure modes visible.

Alternatives considered:
- Treat stitched artifacts as an implementation detail and leave reporting unchanged: rejected because this would understate a known source of localization error.

## Risks / Trade-offs

- [Stitched seams can reduce localization quality around the waterline] -> Keep failure-mode reporting explicit and preserve visibility/confidence outputs so uncertain joints remain inspectable.
- [Single-stream input loses information that separate raw views might preserve] -> Accept this as a baseline simplification and keep raw file metadata available for future multi-view changes.
- [Older notes or scripts may still mention dual-view assumptions] -> Update spec, docs, manifests, and task lists together so the repository tells one consistent story.
- [Future contributors may misread optional raw paths as required inputs] -> Make the canonical-versus-optional distinction explicit in the modified requirements.

## Migration Plan

1. Update the OpenSpec requirement blocks so stitched video is the only required baseline input.
2. Align README, annotation guidance, manifest expectations, and evaluation notes with the stitched-only baseline.
3. Simplify any remaining implementation hooks or comments that still imply synchronized dual-view processing is required.
4. Re-run baseline documentation or validation outputs only if the contract change affects generated examples.

## Open Questions

- Should future multi-view experiments be proposed as a separate capability extension, or as an optional variant under the same capability?
- Do we want to formally deprecate `raw_above_path` and `raw_under_path` later, or keep them indefinitely as optional provenance fields?
