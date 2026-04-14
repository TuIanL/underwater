## MODIFIED Requirements

### Requirement: Canonical dual-view source data
The project SHALL treat one stitched composite video as the canonical source data for breaststroke keypoint localization. The baseline manifest MUST require the stitched clip path, athlete identifier, and session identifier for each clip. Raw above-water and underwater videos MAY be recorded as optional provenance metadata when available, but they MUST NOT be required inputs for baseline training or evaluation.

Optional transfer from Phase 1 video pretraining into the 2D localization pipeline MUST use an explicit bridge mechanism or bridge artifact. The localization baseline MUST NOT assume that a Phase 1 video-pretraining checkpoint is directly compatible with the 2D heatmap model.

#### Scenario: Preparing a clip for training or evaluation
- **WHEN** a clip is prepared for the localization pipeline
- **THEN** the dataset manifest records the stitched clip path together with the athlete identifier and session identifier as the required baseline fields

#### Scenario: Raw camera files also exist for a clip
- **WHEN** a clip also has separate above-water or underwater source files
- **THEN** those files are treated as optional metadata and do not change the stitched clip's role as the canonical baseline input

#### Scenario: Localization training wants to reuse Phase 1 information
- **WHEN** the team opts to transfer knowledge from a Phase 1 video-pretraining checkpoint into the 2D localization path
- **THEN** it does so through an explicit bridge strategy or bridge artifact rather than direct checkpoint loading
