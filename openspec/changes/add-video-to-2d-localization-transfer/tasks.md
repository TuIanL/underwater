## 1. Transfer Strategy

- [ ] 1.1 Compare explicit bridge options for bringing Phase 1 video-pretrained representations into the 2D localization model
- [ ] 1.2 Select an initial bridge strategy and define the artifacts it needs

## 2. Checkpoint And Config Semantics

- [ ] 2.1 Define distinct checkpoint metadata for video-pretraining, bridge, and localization artifacts
- [ ] 2.2 Define how localization configs opt into the bridge without implying direct checkpoint compatibility

## 3. Validation Plan

- [ ] 3.1 Design a small spike experiment to test whether the bridge improves held-out localization quality
- [ ] 3.2 Define the comparison baseline and success criteria for adopting the bridge
