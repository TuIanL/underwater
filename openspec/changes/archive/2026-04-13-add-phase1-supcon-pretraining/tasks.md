## 1. Video Ingestion

- [x] 1.1 Add canonical raw-video parsing rules for `data/raw/videos/<athlete_id>/<session_id>/<stroke>_<take>` and validate supported single-stroke filename tokens
- [x] 1.2 Implement a normalized SupCon video index builder that records source path, athlete, session, stroke label, and validation status for each discovered video
- [x] 1.3 Surface invalid and mixed-stroke videos in ingestion output so default Phase 1 training excludes them without silent guessing

## 2. SupCon Data And Model

- [x] 2.1 Implement a clip-level Phase 1 dataset that reads the normalized video index, samples temporal clips, and returns `(view_1, view_2, label)` batches
- [x] 2.2 Add water-aware spatial and temporal augmentation utilities for independent dual-view construction
- [x] 2.3 Implement a video encoder plus projection head that accepts `B x C x T x H x W` clips and produces reusable encoder features with normalized projection embeddings
- [x] 2.4 Implement numerically stable supervised contrastive loss using stroke labels as positive-pair supervision

## 3. Training Integration

- [x] 3.1 Add a dedicated Phase 1 training loop and config path that consume the normalized video index without changing the existing localization training flows
- [x] 3.2 Add a separate CLI entrypoint for SupCon pretraining and keep the current supervised and semi-supervised keypoint commands unchanged
- [x] 3.3 Save Phase 1 checkpoints and metrics in a form that preserves encoder reuse for later downstream fine-tuning

## 4. Verification

- [x] 4.1 Add tests or deterministic fixtures for canonical path parsing, stroke-label extraction, and invalid-video classification
- [x] 4.2 Run ingestion against the repository's current raw videos and confirm that valid single-stroke videos are indexed while ambiguous videos are flagged
- [x] 4.3 Run a smoke test of the Phase 1 pretraining path to verify clip batching, loss computation, and checkpoint writing
