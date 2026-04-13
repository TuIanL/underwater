## Context

The repository currently mixes multiple environment assumptions:

- TOML configs under `configs/` use relative paths such as `data/...` and `artifacts/...`.
- Generated CSV and JSON artifacts also store relative paths that implicitly assume the repository root as their base.
- Runtime code often resolves those paths from the process current working directory, which means the same command succeeds from the repository root but fails from a subdirectory.
- The documented setup flow uses `uv venv` plus `uv pip install -e .`, while the repository also commits a `uv.lock` file that suggests a stricter synchronization model.
- The helper script `scripts/launch_annotation_gui.sh` bypasses the project environment and invokes `/usr/bin/python3` with `PYTHONPATH=src`.
- Optional tools such as `ffprobe` change runtime behavior, but the environment contract around them is not clearly documented.

This change explores how to turn those implicit assumptions into one explicit local-development contract without rewriting the core modeling workflow.

## Goals / Non-Goals

**Goals:**
- Make supported CLI workflows behave consistently regardless of the current working directory inside the repository.
- Define one official local setup and invocation path for contributors and lab machines.
- Keep committed configs and generated project artifacts portable between machines.
- Make helper scripts and optional tooling expectations explicit.

**Non-Goals:**
- Replacing `uv` with another environment manager.
- Redesigning the training, annotation, or evaluation pipelines beyond environment and path handling.
- Forcing all user-supplied external assets to live inside the repository.
- Finalizing every implementation detail before the exploration is complete.

## Decisions

### 1. Split persisted paths into explicit classes before defining resolution rules

Not every path in the system has the same semantics. The exploration should treat them as separate classes:

| Path class | Examples | On-disk representation | Resolution rule |
|------------|----------|------------------------|-----------------|
| Repository-root-relative artifact paths | `configs/*.toml` values such as `data/...` and `artifacts/...`, manifest source paths when they live inside the repository, annotation index `annotation_path` | Repository-relative when inside the workspace | Resolve from the repository root |
| Frame-root / image-root-relative asset references | annotation `image_path`, unlabeled frame index `image_path`, prediction `image_path` | Relative to the configured `frame_root` / `image_root` | Resolve from the corresponding root argument or config field |
| External source-asset paths | raw video files discovered from `manifest init`, optional user-supplied export destinations outside the workspace | Absolute if outside the repository | Use as-is |
| Explicit absolute paths | any user-provided absolute path | Absolute | Use as-is |

This classification lets the project preserve portability for committed configs and generated internal artifacts without breaking legitimate references to videos that live outside the repository.

Alternatives considered:
- Keep a single rule for every path string: rejected because internal artifacts and external source assets have different portability needs.
- Keep current working-directory-based resolution: rejected because it is brittle and already causes reproducible failures.
- Rewrite all stored paths to absolute machine-specific paths: rejected because it harms portability for committed configs and repo-managed derived data.

### 2. Treat the repository root as the canonical base for repository-root-relative paths

Existing configs and manifest / annotation artifacts that use repository-relative storage already behave as if `data/...` and `artifacts/...` are rooted at the repository, not at the config file location. The design should preserve that contract and make it explicit for repository-root-relative paths, while keeping `image_path` fields anchored to `frame_root` / `image_root`.

Alternatives considered:
- Resolve config values relative to the config file: rejected because current configs under `configs/` would incorrectly map `data/...` to `configs/data/...`.
- Resolve generated paths relative to the file that stores them: rejected because existing CSV and JSON artifacts were not authored with that invariant.

### 3. Normalize paths at read/load boundaries rather than relying on scattered fallback checks

The project should resolve project-managed relative paths once, near config loading and dataset artifact ingestion, instead of relying on ad hoc `Path.exists()` checks in downstream callers. This keeps the path contract centralized and easier to verify.

Alternatives considered:
- Patch individual commands separately: rejected because it would leave inconsistent behavior across training, inference, evaluation, and data tooling.
- Keep best-effort fallback logic inside each dataset loader: rejected because it hides the real path contract and is easy to miss in future code.

### 4. Define repository-root detection and failure behavior explicitly

The project should determine the repository root from the current working directory by walking upward to the nearest ancestor containing `pyproject.toml`. Supported relative-path workflows therefore assume the process starts inside the repository. If a repository-managed relative path is used and no repository root can be found, the command should fail with a clear diagnostic instead of silently guessing.

This deliberately narrows the support boundary: the project should work from the repository root and from subdirectories inside the repository, but it does not need to promise that a command launched outside the repository can reconstruct repository context from an arbitrary absolute config path.

Alternatives considered:
- Infer the repository root independently from each config or artifact path: possible, but it increases complexity and creates multiple competing anchors.
- Fall back to the process current working directory when root detection fails: rejected because it recreates the original ambiguity.

### 5. Standardize the supported workflow on `uv sync` and `uv run`

The repository already ships a `uv.lock` file, and `uv run swim-pose ...` works without requiring a pre-activated shell. Using `uv sync` plus `uv run` gives a more reproducible and less stateful workflow than assuming an activated virtual environment.

The support matrix should be:

```text
Primary supported workflow
- uv sync --locked
- uv run swim-pose ...
- invocation from the repository root or any subdirectory inside the repository

Compatible but not primary
- source .venv/bin/activate
- swim-pose ... from inside the repository

Not part of the supported contract
- bare swim-pose ... in a fresh shell with no activated environment
- commands launched outside the repository that rely on repository-managed relative paths
```

Alternatives considered:
- Continue documenting `uv venv` plus `uv pip install -e .` as the primary flow: rejected because it depends on shell state and does not emphasize lockfile synchronization.
- Require direct execution through `.venv/bin/python`: workable but less ergonomic and less aligned with the existing `uv` tooling already in the project.

### 6. Preserve compatibility for existing repository-relative artifacts and frame-root-relative image references, but not for ambiguous caller-relative externals

Historical generated data such as annotation indices that already store repository-relative `annotation_path` values should remain valid after the change. Historical `image_path` values that are already relative to `frame_root` / `image_root` should also remain valid after the change. Historical absolute paths should continue to work unchanged.

The ambiguous case is historical persisted paths that are relative but refer to locations outside the repository because they were captured from the caller's working directory. The project should not silently reinterpret those values as repository-relative. Instead, the design should treat them as legacy ambiguous data: regeneration is acceptable unless a clear migration rule emerges.

Alternatives considered:
- Automatically reinterpret every historical relative path as repository-relative: rejected because it can corrupt references to external source assets.
- Require all historical artifacts to be regenerated regardless of type: rejected because existing repository-relative annotation and frame artifacts are already compatible with the intended contract.

### 7. Keep the annotation wrapper, but make it use the project-managed interpreter

The browser-annotation wrapper script should remain available as a convenience entrypoint, but it should launch through `uv run` inside the project rather than through `/usr/bin/python3`. This keeps the ergonomic shortcut while removing interpreter drift.

Alternatives considered:
- Keep `/usr/bin/python3` plus `PYTHONPATH=src`: rejected because it bypasses the locked environment and will diverge as dependencies grow.
- Remove wrapper scripts entirely: rejected because the project still benefits from a simple launcher for local labeling.

### 8. Add a migration tool for legacy caller-relative manifest paths

Ambiguous historical relative paths that point outside the repository should be handled with an explicit manifest migration command rather than through silent runtime guessing. The migration tool should accept the manifest plus the original working-directory base used when those rows were written, then rewrite in-repository videos to repository-relative form and out-of-repository videos to absolute form.

Alternatives considered:
- Continue with documentation-only regeneration guidance: rejected because some historical manifests can be migrated deterministically when the original working-directory base is known.
- Silently rebase every relative manifest source path to the repository root: rejected because it would mis-handle external video locations.

### 9. Treat optional system tools as explicit optional capabilities

System tools such as `ffprobe` can remain optional, but the project should clearly state what changes when they are absent. Explore mode should preserve graceful fallback behavior while making the limitation visible in documentation and validation plans.

Alternatives considered:
- Make `ffprobe` a hard requirement: not necessary for all workflows.
- Ignore the distinction and leave behavior implicit: rejected because it increases environment confusion during onboarding and debugging.

## Risks / Trade-offs

- [Repository-root path resolution couples project-managed artifacts to the workspace layout] -> Accept this for committed configs and generated project data, while continuing to allow explicit absolute paths where appropriate.
- [The path taxonomy introduces more rules to explain] -> Counter this with one concise table in design/docs and by centralizing path normalization in one helper layer.
- [Cross-cutting path changes may miss one reader or writer] -> Build an explicit inventory of path-bearing fields and validate representative workflows before implementation.
- [Caller-relative historical external paths may remain ambiguous] -> Treat them as migration candidates and document regeneration as the safe fallback.
- [Switching documentation to `uv sync` / `uv run` may surprise contributors used to activating `.venv`] -> Preserve the reasoning in docs and keep compatibility notes if needed during the transition.
- [Wrapper scripts may need extra care on machines where `uv` is not installed yet] -> Treat `uv` as part of the supported local tooling and document that dependency once.

## Migration Plan

1. Inventory all path-bearing fields across TOML configs, manifests, annotation files, derived indices, and helper scripts.
2. Encode the path and environment contract in proposal/spec/design artifacts.
3. Update the affected loaders, scripts, and docs to follow the chosen contract.
4. Validate the resulting workflow from the repository root, from a subdirectory, and from a fresh shell with no activated virtual environment.
5. Validate compatibility against existing repository-relative generated artifacts and identify any ambiguous historical caller-relative externals that need regeneration guidance.
6. Verify optional-tool behavior such as `ffprobe` fallback remains explicit.

## Open Questions

- None at this stage.
