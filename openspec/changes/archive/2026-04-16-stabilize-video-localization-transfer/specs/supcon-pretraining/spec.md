## MODIFIED Requirements

### Requirement: Water-aware spatiotemporal augmentation
The Phase 1 dataset MUST support strong augmentations that preserve stroke semantics while reducing shortcut signals from underwater conditions. The augmentation set SHALL include temporal sampling variation, spatial appearance perturbations such as crop, color jitter or grayscale, blur, and a configurable temporal tube-masking operator that masks the same spatial region across a short contiguous span of frames. The tube-masking operator MUST be able to be disabled by configuration so experiments can roll back to the prior augmentation policy without code changes.

#### Scenario: Two views are constructed from one clip
- **WHEN** the dataset prepares a positive pair for contrastive learning
- **THEN** each view applies an independent augmentation pipeline so the pair is not pixel-identical, and either view MAY independently include tube masking

#### Scenario: Pool color differs across sessions
- **WHEN** videos from different pools or lighting conditions are sampled
- **THEN** the augmentation pipeline reduces reliance on color and background cues without changing the stroke label

#### Scenario: Splash-like occlusion augmentation is enabled
- **WHEN** a training view samples tube masking
- **THEN** the augmentation masks one spatial region across consecutive frames within that view while leaving the remainder of the clip available for temporal reasoning

#### Scenario: Conservative rollback disables tube masking
- **WHEN** tube masking is disabled in the SupCon configuration
- **THEN** Phase 1 pretraining falls back to the prior crop, color, grayscale, and blur augmentation behavior without requiring code changes

### Requirement: Dedicated Phase 1 configuration and entrypoint
The system SHALL provide a dedicated configuration path and training entrypoint for Phase 1 supervised contrastive pretraining. This entrypoint MUST remain separate from the existing keypoint localization training commands and MUST expose the selected video-backbone, conservative-upgrade feature toggles, and training-stability settings needed for experiment-grade runs. Conservative-upgrade toggles MUST allow newly introduced augmentation or memory-related behaviors to be disabled independently so the team can bisect regressions without restoring old code.

#### Scenario: User launches Phase 1 training
- **WHEN** a user selects the Phase 1 SupCon training command with a valid config
- **THEN** the system starts the video-based pretraining workflow with the configured video encoder and training settings without changing the behavior of the existing localization commands

#### Scenario: User still trains the localization baseline
- **WHEN** a user runs the existing supervised or semi-supervised localization training commands
- **THEN** those commands continue to operate on the frame-level localization pipeline as before

#### Scenario: Stage-specific conservative feature is disabled
- **WHEN** a SupCon config disables tube masking or related conservative-upgrade settings
- **THEN** the training loop applies the remaining enabled settings and leaves disabled behaviors inactive without requiring source changes
