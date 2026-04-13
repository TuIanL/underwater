# local-environment-setup Specification

## Purpose
TBD - created by archiving change stabilize-local-environment-setup. Update Purpose after archive.
## Requirements
### Requirement: Repository-managed relative paths resolve from the repository root

The project SHALL interpret relative paths in committed experiment configs and generated dataset artifacts against the repository root rather than the process current working directory. Explicit absolute paths MAY still be used and MUST remain unchanged.

#### Scenario: Training is launched from a subdirectory

- **WHEN** a user runs a training or inference command from a subdirectory and passes a config that contains `data/...` or `artifacts/...` paths
- **THEN** the command resolves those relative paths as repository-root-relative and reads or writes the intended files

#### Scenario: A generated index contains relative file references

- **WHEN** a loader reads a manifest row, annotation index row, or unlabeled frame index row that stores project-generated relative paths
- **THEN** those paths resolve to the same files regardless of the caller's current working directory

#### Scenario: An explicit absolute path is supplied

- **WHEN** a user provides an absolute checkpoint, config, data, or output path
- **THEN** the project uses that absolute path unchanged rather than rebasing it to the repository root

### Requirement: Persisted paths distinguish repository-root-relative assets, frame-root-relative assets, and external source assets

The project SHALL persist repository-root-relative assets as repository-relative when those assets live inside the repository workspace. The project SHALL persist `image_path`-style frame references relative to the configured `frame_root` / `image_root`. The project SHALL persist external source-asset paths as absolute when those assets live outside the repository workspace.

#### Scenario: A manifest is created from repository-local videos

- **WHEN** `manifest init` scans videos that live under the repository workspace
- **THEN** the generated manifest stores those source paths in repository-relative form

#### Scenario: A manifest is created from videos outside the repository

- **WHEN** `manifest init` scans videos that live outside the repository workspace
- **THEN** the generated manifest stores those source paths in absolute form so later commands do not depend on the original caller working directory

#### Scenario: A derived artifact references repository-managed frames or annotations

- **WHEN** the project writes annotation JSON, annotation index CSV, unlabeled frame index CSV, or prediction rows for files under the repository workspace
- **THEN** annotation-path-style references use repository-relative paths while `image_path` references stay relative to the configured `frame_root` / `image_root`

### Requirement: Supported local setup uses a reproducible `uv` workflow

The project SHALL define one supported local setup workflow based on the repository's `uv` environment and lockfile, and documented command examples MUST NOT depend on prior manual shell activation to succeed. `uv sync --locked` and `uv run` SHALL be the primary documented workflow.

#### Scenario: A contributor sets up the project from a fresh clone

- **WHEN** the contributor follows the documented setup steps
- **THEN** the environment is synchronized to the committed dependency lock and the CLI can be invoked through the documented `uv` workflow

#### Scenario: A contributor runs a documented command in a fresh shell

- **WHEN** the contributor uses the documented invocation pattern for a project command
- **THEN** the command succeeds without assuming that `swim-pose` is already on the shell `PATH` from an activated virtual environment

#### Scenario: A contributor prefers an activated virtual environment

- **WHEN** the contributor activates the project's `.venv` manually and runs a command from inside the repository
- **THEN** the command MAY remain compatible, but the documentation still treats the `uv` workflow as the primary supported path

### Requirement: Missing repository context fails explicitly

If a command needs to resolve a repository-managed relative path and cannot determine the repository root from the current working directory, the project MUST fail with a clear diagnostic rather than guessing a fallback base directory.

#### Scenario: A repository-relative workflow is launched outside the repository

- **WHEN** a user launches a command from outside the repository while relying on repository-managed relative paths
- **THEN** the command exits with an error that explains the missing repository context

#### Scenario: Historical ambiguous external paths are encountered

- **WHEN** the project reads a historical persisted relative path whose meaning cannot be unambiguously classified as repository-managed
- **THEN** the project surfaces migration guidance through the manifest path migration command or requires regeneration instead of silently rebasing the path to the repository root

### Requirement: Helper scripts and optional tooling behave explicitly

Project helper scripts SHALL use the same managed Python environment as the supported CLI workflow, and optional system tools that change command behavior MUST be documented together with their fallback or failure mode.

#### Scenario: A helper script launches a local web UI

- **WHEN** a contributor uses a provided wrapper script for annotation or prediction tooling
- **THEN** the script runs against the project's managed Python dependencies instead of an unrelated system interpreter

#### Scenario: Media probing tools are unavailable

- **WHEN** a command depends on video metadata and `ffprobe` or related system tooling is not installed
- **THEN** the project either uses its documented fallback path or surfaces a clear diagnostic that explains the limitation

