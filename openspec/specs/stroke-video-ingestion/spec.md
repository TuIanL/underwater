# stroke-video-ingestion Specification

## Purpose
TBD - created by archiving change add-phase1-supcon-pretraining. Update Purpose after archive.
## Requirements
### Requirement: Canonical raw video layout
The system SHALL treat `data/raw/videos/<athlete_id>/<session_id>/` as the canonical repository-managed drop location for Phase 1 raw videos. Each Phase 1 candidate video filename MUST begin with a recognized single-stroke token and MAY include a take suffix, for example `蛙_01.mp4`.

#### Scenario: User adds a valid single-stroke video
- **WHEN** a user places `data/raw/videos/yaojunji/2026-04-13/蛙_01.mp4` in the repository
- **THEN** the ingestion flow recognizes `yaojunji` as the athlete identifier, `2026-04-13` as the session identifier, and `蛙` as the stroke label

#### Scenario: User places a file outside the canonical layout
- **WHEN** a video is stored outside `data/raw/videos/<athlete_id>/<session_id>/`
- **THEN** the ingestion flow does not treat that file as a valid Phase 1 training input

### Requirement: Normalized video index generation
The system SHALL provide an ingestion flow that scans the canonical raw video layout and writes a normalized repository-managed video index for Phase 1. Each accepted index row MUST record the source video path, athlete identifier, session identifier, stroke label, and validation status.

#### Scenario: Canonical layout is scanned
- **WHEN** ingestion runs over a directory that contains valid Phase 1 videos
- **THEN** the system writes an index that includes one row per discovered video with the parsed metadata and source path

#### Scenario: A user inspects parsed metadata
- **WHEN** a user opens the generated index after ingestion
- **THEN** the user can verify which athlete, session, stroke label, and validation status were assigned to each source video

### Requirement: Ambiguous and invalid videos are surfaced explicitly
The ingestion flow MUST classify videos whose filenames do not begin with a recognized single-stroke token as invalid for Phase 1 supervision. Videos that represent multi-stroke compilations or otherwise ambiguous labels MUST be surfaced in the generated index or companion report and MUST be excluded from the default training set.

#### Scenario: Mixed-stroke compilation is present
- **WHEN** ingestion encounters a file such as `四式合集.mp4`
- **THEN** the file is marked as invalid or mixed-stroke and is excluded from the default Phase 1 training index

#### Scenario: Unknown filename token is present
- **WHEN** ingestion encounters a file whose stem does not begin with one of the supported single-stroke tokens
- **THEN** the system records the file as invalid rather than guessing a stroke label

### Requirement: Ingestion preserves reproducibility
The normalized Phase 1 video index SHALL be sufficient for downstream training to run without reinterpreting the raw directory structure on every batch fetch. Training configs MUST be able to reference the generated index directly.

#### Scenario: Training starts from a generated index
- **WHEN** a Phase 1 training job is launched after ingestion has completed
- **THEN** the training job reads the generated video index instead of reparsing the raw filesystem for every sample

#### Scenario: Raw folders are reorganized after indexing
- **WHEN** a user changes raw folders after an index has been generated
- **THEN** the system requires a new ingestion run before the updated files are reflected in future training jobs

