# Known Failure Modes And Open Questions

## Expected Failure Modes

- Waterline seam causes unstable head, shoulder, and wrist localization when only stitched footage is reviewed.
- Splash-heavy frames reduce confidence on wrists, toes, and heels.
- Side-view overlap can cause left/right confusion on ankles and knees if annotators do not follow anatomical identity rules.
- Professionally trained swimmers may narrow visual diversity and produce optimistic generalization estimates.
- Weak pseudo-label filtering can reinforce early model mistakes instead of correcting them.

## Current Experiment Notes

- Current reviewed seed set contains 6 labeled frames and 6 `no_swimmer` frames from one stitched breaststroke clip, so all quantitative results are preliminary and do not represent held-out athlete generalization.
- The supervised baseline currently outperforms both semi-supervised variants on the labeled subset. The temporal-constraint variant is slightly better than the plain semi-supervised run, but it still trails the supervised baseline on overall error and temporal stability.
- The hardest joints remain the distal lower-body landmarks, especially `right_ankle`, `right_heel`, `right_toe`, `left_ankle`, and `left_toe`.
- Occluded-point behavior is still unstable. The semi-supervised run improved the tiny occluded subset numerically, but regressed most visible-joint metrics.
- Temporal jitter is still high on stitched-video predictions, which suggests seam transitions and lower-body ambiguity are dominating the motion instability signal.

## Open Questions

- How much stitched-seam distortion can the baseline tolerate before lower-body landmarks become unreliable?
- Should separate raw camera files remain provenance only, or be revisited in a future multi-view change?
- Is a breathing-specific point such as `chin` necessary in a future change?
