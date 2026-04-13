## Why

The current project environment works only when several implicit assumptions line up: commands are launched from the repository root, the correct virtual environment is already activated, and helper scripts happen to use a compatible interpreter. Relative paths in configs and generated dataset artifacts are therefore brittle, which makes the workflow hard to reproduce across shells, directories, and machines.

## What Changes

- Define a stable contract for how project-managed relative paths are interpreted across configs, manifests, annotation data, and derived indices.
- Classify persisted paths into repository-managed artifact references versus external source-asset references, with explicit backward-compatibility expectations for existing generated data.
- Standardize the supported local setup and command invocation workflow around the repository's `uv`-managed environment and lockfile.
- Define the supported invocation matrix, including the primary `uv` workflow, compatibility expectations for activated virtual environments, and clear non-goals for unsupported launch modes.
- Align helper scripts with the same interpreter and dependency model as the main CLI.
- Add a manifest migration command for legacy caller-relative source-video paths that actually point outside the repository.
- Clarify which external media tools are optional and how commands behave when those tools are unavailable.

## Capabilities

### New Capabilities
- `local-environment-setup`: Defines how developers install, invoke, and run the project locally, including path resolution rules for project-managed artifacts.

### Modified Capabilities
- None.

## Impact

- Affects CLI-adjacent modules under `src/swim_pose/` that read config, manifest, annotation, and dataset paths.
- Affects local helper scripts under `scripts/`.
- Affects contributor-facing setup and command documentation in `README.md`.
- Affects expectations around `uv.lock`, shell activation, and optional tools such as `ffprobe` / `ffmpeg`.
