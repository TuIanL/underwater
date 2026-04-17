## 1. YOLO Baseline Scaffolding

- [x] 1.1 Add and pin the external YOLO pose dependency and introduce project-owned config fields for the new default baseline.
- [x] 1.2 Implement dataset conversion from the current annotation/index format into the pose-training format required by the YOLO baseline while preserving 18-keypoint schema metadata.
- [x] 1.3 Add project-owned training and inference entrypoints that run the YOLO baseline without exposing vendor-specific artifact contracts to downstream tools.

## 2. Schema Adaptation And Underwater Fine-Tuning

- [x] 2.1 Implement the schema-adaptation layer that maps pretrained pose priors into the project's 18-keypoint contract, including explicit handling for `neck`, `heel`, and `toe`.
- [x] 2.2 Translate baseline inference outputs back into the project's frame-keyed raw prediction format with confidence and visibility-related fields.
- [x] 2.3 Add underwater-domain adaptation presets for splash, blur, seam artifacts, and self-occlusion, then document how they are enabled in baseline experiments.

## 3. Temporal Postprocessing And Evaluation

- [x] 3.1 Implement a confidence-aware temporal postprocessing stage with an explicit enable/disable toggle and stable artifact naming.
- [x] 3.2 Extend prediction export and downstream consumers so raw and filtered trajectories can be stored side by side for the same frame sequence.
- [x] 3.3 Replace the current motion-as-jitter reporting with raw-versus-filtered stability evaluation that separates prediction flicker from real swimmer motion.

## 4. Migration, Verification, And Documentation

- [x] 4.1 Reclassify the current custom heatmap, SupCon, and bridge paths as research baselines in configs, metadata, and user-facing docs.
- [x] 4.2 Add regression coverage for dataset conversion, 18-keypoint export contracts, and temporal postprocessing toggle behavior.
- [x] 4.3 Run the first side-by-side baseline comparison and record rollout plus rollback criteria for adopting the YOLO path as the default localization baseline.
