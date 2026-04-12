# Breaststroke 18-Point Annotation Guide

## Purpose

This guide defines the canonical annotation rules for the breaststroke localization dataset. The goal is to preserve anatomically meaningful joint identities across waterline transitions, splash, overlap, and temporary occlusion.

## View Policy

- Use the stitched/composited video as the canonical labeling source for the phase-one baseline and mark labeled frames as `source_view = stitched`.
- If separate above-water or underwater files also exist, keep them as provenance only; do not switch the annotation workflow away from the stitched timeline.
- Keep the athlete's anatomical left and right identities fixed across all frames, even if the near-side limb appears on the opposite side of the image.

## Keypoint Definitions

| Keypoint | Rule |
| --- | --- |
| `nose` | Tip of the nose or closest visible nasal profile point |
| `neck` | Midpoint at the base of the neck between the shoulders |
| `left_shoulder` / `right_shoulder` | Center of the glenohumeral joint |
| `left_elbow` / `right_elbow` | Center of the elbow hinge |
| `left_wrist` / `right_wrist` | Center of the wrist joint |
| `left_hip` / `right_hip` | Center of the hip joint projected to the body contour |
| `left_knee` / `right_knee` | Center of the knee hinge |
| `left_ankle` / `right_ankle` | Center of the ankle joint |
| `left_heel` / `right_heel` | Posterior midpoint of the heel |
| `left_toe` / `right_toe` | Most distal visible toe or forefoot tip |

## Visibility States

- `2`: directly visible and localizable
- `1`: not directly visible, but inferable from anatomy and nearby context
- `0`: not visible and not reliably inferable

## Frame Status Rules

- `labeled`: this frame is ready to be used as supervised training data
- `pending`: this frame has not been finished yet
- `review`: this frame was labeled, but should be checked again because it is difficult or ambiguous
- `no_swimmer`: the swimmer is not in the frame at all, so all 18 keypoints must stay empty

Do not silently skip early or empty frames. If the swimmer has not entered the view yet, mark the frame as `no_swimmer` so the dataset keeps an explicit record of why the JSON is blank.

## Waterline Rules

- Do not switch identity because a limb crosses the waterline seam in a stitched preview.
- If the seam or reflection hides the exact joint center but the joint can still be estimated, annotate the coordinate and mark visibility `1`.
- If the joint is fully indeterminate at the seam, leave the coordinate empty and mark visibility `0`.

## Heel And Toe Rules

- Place `heel` on the back of the foot, not the ankle.
- Place `toe` on the most distal toe or forefoot tip visible in the frame.
- When the foot is heavily foreshortened, place the point where the distal foot contour is best supported and downgrade visibility if needed.

## Difficult Cases

- Splash hides the joint but body geometry is clear: annotate with visibility `1`.
- Limb leaves the frame: visibility `0`.
- Swimmer not in frame yet: set frame status to `no_swimmer`, leave all keypoints empty.
- Two limbs overlap: keep anatomical left/right labels, even if one joint center must be inferred.
- Motion blur with no clear endpoint: visibility `0` unless the location is still defensible from adjacent structure.

## Recommended Review Checklist

- Are left and right identities anatomically correct?
- Are heel and toe points separated from ankle points?
- Do visibility labels match the actual certainty of the frame?
- Are hard frames around waterline crossings and overlap represented in the labeled set?
