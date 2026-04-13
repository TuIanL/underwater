## Context

The repository currently supports frame-level keypoint localization from stitched swim frames and annotations. Its training entrypoints, dataset classes, and manifests are built around image tensors plus heatmap supervision, which does not match Phase 1 supervised contrastive pretraining. The new workflow must let users place raw videos into a repository-managed folder layout, automatically infer stable metadata, exclude ambiguous samples, and then train a video encoder on temporal clips without disrupting the existing localization pipeline.

The most important constraint is data hygiene. SupCon benefits from simple automation, but it also depends on label consistency: each training sample must represent a single stroke class, and the metadata parser must not confuse archive folders, athlete names, or session names for one another. The design therefore needs both a strict ingestion contract and a separate training path.

## Goals / Non-Goals

**Goals:**
- Define a fixed raw-video folder and naming convention that supports drop-in ingestion.
- Generate a normalized, repository-managed video index that records athlete, session, stroke label, source path, and validation status for each candidate training video.
- Add a dedicated Phase 1 SupCon training path that consumes temporal clips, returns dual augmented views, and trains an encoder plus projection head with supervised contrastive loss.
- Preserve the existing keypoint localization training commands, configs, and manifests.
- Make mixed-stroke or ambiguous videos visible to users and easy to exclude before training.

**Non-Goals:**
- Replacing the current frame-level localization manifests or training entrypoints.
- Solving downstream swim-style classification fine-tuning in this change.
- Supporting arbitrary folder layouts or inferring reliable labels from free-form archive names.
- Automatically splitting a multi-stroke compilation video into stroke-specific segments.

## Decisions

### 1. Use a strict repository-managed raw video convention

Raw videos for Phase 1 will live under a fixed layout such as `data/raw/videos/<athlete_id>/<session_id>/<stroke>_<take>.mp4`. The folder path provides `athlete_id` and `session_id`; the filename stem provides `stroke_label` plus an optional take suffix.

Why:
- It makes the user workflow simple: copy videos into one known location.
- It removes the need to interpret package names or ad hoc Chinese archive folders as identity metadata.
- It keeps label extraction deterministic and testable.

Alternatives considered:
- Reusing the current generic manifest discovery against arbitrary directories. Rejected because the current inference logic can treat archive folders as athlete/session metadata.
- Requiring a hand-authored CSV manifest. Rejected because it adds manual bookkeeping and is exactly the overhead this change is trying to remove.

### 2. Insert an explicit ingestion/index build step before training

The system will scan the fixed raw-video directory and emit a normalized video index for SupCon rather than having the training loop interpret the filesystem directly on every run.

Why:
- A generated index is inspectable, versionable, and easy to debug when labels look wrong.
- Validation status can be recorded once and reused across training runs.
- It matches the repository's current pattern of manifest- and index-based pipelines.

Alternatives considered:
- Training directly from the directory tree without an intermediate index. Rejected because it hides validation failures and makes reproducibility worse.

### 3. Treat single-stroke filename prefixes as the only valid Phase 1 label source

The ingestion flow will accept only a bounded set of stroke labels, initially `蛙`, `仰`, `蝶`, and `自`, derived from the video filename stem. Files whose stems do not begin with a recognized single-stroke token will be marked invalid or mixed and excluded from the default training index.

Why:
- It gives SupCon a stable supervision source with no extra labeling UI.
- It creates a clear escape hatch for ambiguous videos such as four-stroke compilations.

Alternatives considered:
- Inferring labels from parent folders or notes. Rejected because folder semantics are inconsistent across existing archives.
- Accepting free-form labels and normalizing later. Rejected because silent normalization errors would be costly to debug.

### 4. Add a separate SupCon training path instead of overloading existing supervised training

Phase 1 pretraining will introduce dedicated dataset, model, loss, config, and CLI entrypoints. The existing `train supervised` and `train semisupervised` flows will continue to serve keypoint localization.

Why:
- The existing pipeline operates on `image -> heatmaps/visibility`, which is a different task contract from `clip -> embedding`.
- Keeping the paths separate reduces regression risk and preserves team confidence in the current pose work.

Alternatives considered:
- Adding mode flags to the current supervised training function. Rejected because it would blur incompatible tensor shapes, outputs, and losses inside one entrypoint.

### 5. Standardize on a 5D temporal clip interface and a projection-head-based encoder

The SupCon dataset will produce clip tensors shaped `C x T x H x W`, and batching will yield `B x C x T x H x W`. The model will consist of a video encoder that outputs pooled features and a two-layer projection head that produces L2-normalized embeddings for contrastive training.

Why:
- This matches the intended Phase 1 representation-learning objective.
- It keeps the implementation compatible with torchvision video backbones and future encoder upgrades.
- It cleanly separates reusable encoder features from training-only projection embeddings.

Alternatives considered:
- Reusing the 2D frame encoder with only image-level augmentation. Rejected because it underserves the temporal nature of swimming stroke patterns.

### 6. Use validation-driven exclusion for mixed or low-quality inputs

Ingestion will mark each video with a validation status such as valid, mixed-stroke, unknown-label, or unreadable. Training configs will default to consuming only valid single-stroke videos.

Why:
- It prevents low-quality supervision from silently contaminating the contrastive batches.
- It gives users a concrete report of what needs renaming, relocation, or removal.

Alternatives considered:
- Letting training skip failures on the fly. Rejected because users would not get a stable view of dataset quality.

## Risks / Trade-offs

- [Strict folder rules may feel rigid] -> Mitigation: keep the rule simple and document one canonical layout users can follow by copy/paste.
- [Current archive folders contain useful videos that do not match the canonical layout] -> Mitigation: allow a one-time reorganization step or future manifest-import helper, but keep Phase 1 ingestion strict.
- [Class counts may remain small even after ingestion is automated] -> Mitigation: Phase 1 focuses on building a correct and repeatable pipeline first; larger corpora can be added without redesigning the flow.
- [Video decoding can slow down training] -> Mitigation: cache the normalized index, sample clips lazily, and keep clip length and frame stride configurable.
- [Mixed-stroke compilations may be tempting to keep] -> Mitigation: surface them in the ingestion report but exclude them by default so label noise stays explicit.

## Migration Plan

1. Introduce the raw-video convention and normalized SupCon video index alongside the current manifests.
2. Add the SupCon dataset, model, loss, config, and CLI entrypoint behind new files or clearly separate code paths.
3. Validate the ingestion flow against the current repository videos and confirm that ambiguous files are reported instead of silently accepted.
4. Run a small dry run of the SupCon training path using the generated index.
5. Keep the existing localization configs and commands unchanged so rollback is as simple as not using the new SupCon entrypoint.

## Open Questions

- Should the initial canonical stroke tokens remain the Chinese short names (`蛙`, `仰`, `蝶`, `自`) or be normalized to internal English identifiers in the generated index?
- Should ingestion also support a sidecar allowlist or denylist file for temporarily excluding known-problem videos without renaming them?
- How much metadata from the normalized index should be carried into downstream fine-tuning checkpoints for later classification work?
