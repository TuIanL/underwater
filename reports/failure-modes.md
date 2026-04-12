# Known Failure Modes And Open Questions

## Expected Failure Modes

- Waterline seam causes unstable head, shoulder, and wrist localization when only stitched footage is reviewed.
- Splash-heavy frames reduce confidence on wrists, toes, and heels.
- Side-view overlap can cause left/right confusion on ankles and knees if annotators do not follow anatomical identity rules.
- Professionally trained swimmers may narrow visual diversity and produce optimistic generalization estimates.
- Weak pseudo-label filtering can reinforce early model mistakes instead of correcting them.

## Open Questions

- Are both raw streams available and time-aligned for every recorded athlete and session?
- Will the first baseline use a shared model across above-water and underwater views, or separate per-view experiments?
- Is a breathing-specific point such as `chin` necessary in a future change?
- How much synchronization drift is acceptable before a dual-view clip is demoted to single-view usage?

