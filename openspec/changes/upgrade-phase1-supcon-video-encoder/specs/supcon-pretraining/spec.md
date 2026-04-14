## MODIFIED Requirements

### Requirement: Dual-view temporal clip sampling
The Phase 1 pretraining dataset SHALL consume only videos with a valid single-stroke label from the normalized video index. For each sampled training item, the dataset MUST sample one temporal clip, derive two independently augmented views from that same clip, and return those views together with the stroke label.

#### Scenario: Valid video is sampled for training
- **WHEN** the Phase 1 dataset samples a valid row from the normalized video index
- **THEN** both returned views are derived from the same sampled temporal indices and the corresponding stroke label

#### Scenario: Temporal jitter is enabled
- **WHEN** the dataset uses temporal jitter or stride variation while sampling a training item
- **THEN** that temporal variation is applied once to the sampled clip and shared by both augmented views

#### Scenario: Invalid video exists in the index
- **WHEN** the generated index contains rows marked invalid or mixed-stroke
- **THEN** the default Phase 1 training dataset excludes those rows from sampling

### Requirement: Encoder and projection head are trained as a separate pretraining model
The Phase 1 model SHALL accept batched clip tensors shaped `B x C x T x H x W`, encode them with a temporally aware video backbone into pooled feature vectors, and map those features through a projection head that outputs L2-normalized embeddings for contrastive loss. The projection head MUST remain a pretraining-only component, and saved encoder weights MUST be reusable by compatible downstream video consumers without depending on the projection head at inference time.

#### Scenario: Batch enters the pretraining model
- **WHEN** a batch of temporal clips is forwarded through the Phase 1 model
- **THEN** the encoder captures spatiotemporal interactions before global pooling and the model returns pooled encoder features plus normalized projection embeddings

#### Scenario: Pretraining finishes successfully
- **WHEN** a Phase 1 checkpoint is saved for downstream reuse
- **THEN** it preserves reusable video-encoder weights and metadata for compatible downstream video models without requiring the projection head

### Requirement: Dedicated Phase 1 configuration and entrypoint
The system SHALL provide a dedicated configuration path and training entrypoint for Phase 1 supervised contrastive pretraining. This entrypoint MUST remain separate from the existing keypoint localization training commands and MUST expose the selected video-backbone and training-stability settings needed for experiment-grade runs.

#### Scenario: User launches Phase 1 training
- **WHEN** a user selects the Phase 1 SupCon training command with a valid config
- **THEN** the system starts the video-based pretraining workflow with the configured video encoder and training settings without changing the behavior of the existing localization commands

#### Scenario: User still trains the localization baseline
- **WHEN** a user runs the existing supervised or semi-supervised localization training commands
- **THEN** those commands continue to operate on the frame-level localization pipeline as before
