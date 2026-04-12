## 1. Specification Alignment

- [x] 1.1 Update the `breaststroke-keypoint-localization` OpenSpec requirement blocks to make stitched video the canonical baseline input.
- [x] 1.2 Review adjacent requirement text for any remaining dual-view or frame-pair wording and rewrite it for stitched-only processing.

## 2. Documentation Alignment

- [x] 2.1 Update README, annotation guidance, and any baseline notes that still describe synchronized above-water and underwater videos as required inputs.
- [x] 2.2 Clarify in project docs that raw camera files are optional provenance only and that `stitched_path` is the required baseline field.

## 3. Pipeline Alignment

- [x] 3.1 Audit manifest, sampling, frame extraction, and CLI messaging for stale dual-view assumptions and simplify them to the stitched-video baseline where needed.
- [x] 3.2 Audit training, inference, and evaluation comments or config descriptions for cross-view wording that no longer matches the stitched-only method.

## 4. Validation

- [x] 4.1 Run the relevant OpenSpec validation or status checks to confirm the change is apply-ready.
- [x] 4.2 Sanity-check representative generated artifacts or examples so the revised single-video route is described consistently end to end.
