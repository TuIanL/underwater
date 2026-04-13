## 1. Environment Contract Discovery

- [x] 1.1 Inventory all project-managed path fields across TOML configs, manifests, annotation JSON, and derived CSV indices.
- [x] 1.2 Classify persisted paths into repository-managed artifacts, external source assets, and explicit absolute paths.
- [x] 1.3 Identify which historical generated artifacts are already compatible and which caller-relative cases would require regeneration or migration guidance.

## 2. Workflow Decisions

- [x] 2.1 Define the repository-root detection rule and the explicit error behavior when a repository-managed relative path has no valid repository anchor.
- [x] 2.2 Compare the current `uv venv` / `uv pip install -e .` flow with a `uv sync` / `uv run` flow and choose the supported default.
- [x] 2.3 Freeze the supported invocation matrix, including primary, compatibility, and unsupported launch modes.
- [x] 2.4 Decide how helper scripts should locate the project interpreter and whether any wrappers can be replaced by direct CLI commands.

## 3. Validation Planning

- [x] 3.1 Define regression checks for `uv run` from the repository root and from a repository subdirectory.
- [x] 3.2 Define compatibility checks for activated-virtualenv usage and for existing repository-relative generated artifacts.
- [x] 3.3 Define failure or migration checks for ambiguous caller-relative external paths.
- [x] 3.4 Define validation cases for optional external tools such as `ffprobe` so fallback behavior stays explicit after the change.

## 4. Implementation Handoff

- [x] 4.1 Translate the approved path and workflow decisions into concrete code and documentation changes once explore mode is complete.
