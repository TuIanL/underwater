from dataclasses import dataclass


@dataclass(frozen=True)
class KeypointSpec:
    name: str
    group: str
    description: str


KEYPOINT_SPECS = (
    KeypointSpec("nose", "head", "Tip of the nose or closest visible nasal profile point."),
    KeypointSpec("neck", "head", "Midpoint at the base of the neck between the shoulders."),
    KeypointSpec("left_shoulder", "upper", "Center of the left shoulder joint."),
    KeypointSpec("right_shoulder", "upper", "Center of the right shoulder joint."),
    KeypointSpec("left_elbow", "upper", "Center of the left elbow hinge."),
    KeypointSpec("right_elbow", "upper", "Center of the right elbow hinge."),
    KeypointSpec("left_wrist", "upper", "Center of the left wrist joint."),
    KeypointSpec("right_wrist", "upper", "Center of the right wrist joint."),
    KeypointSpec("left_hip", "lower", "Center of the left hip joint."),
    KeypointSpec("right_hip", "lower", "Center of the right hip joint."),
    KeypointSpec("left_knee", "lower", "Center of the left knee hinge."),
    KeypointSpec("right_knee", "lower", "Center of the right knee hinge."),
    KeypointSpec("left_ankle", "lower", "Center of the left ankle joint."),
    KeypointSpec("right_ankle", "lower", "Center of the right ankle joint."),
    KeypointSpec("left_heel", "foot", "Posterior midpoint of the left heel."),
    KeypointSpec("right_heel", "foot", "Posterior midpoint of the right heel."),
    KeypointSpec("left_toe", "foot", "Most distal visible point on the left forefoot."),
    KeypointSpec("right_toe", "foot", "Most distal visible point on the right forefoot."),
)

KEYPOINT_NAMES = tuple(spec.name for spec in KEYPOINT_SPECS)

VISIBILITY_STATES = {
    0: "not_visible",
    1: "inferable",
    2: "visible",
}

FRAME_STATUSES = (
    "pending",
    "labeled",
    "no_swimmer",
    "review",
)

SKELETON_EDGES = (
    ("nose", "neck"),
    ("neck", "left_shoulder"),
    ("neck", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_elbow", "right_wrist"),
    ("neck", "left_hip"),
    ("neck", "right_hip"),
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"),
    ("right_ankle", "right_heel"),
    ("left_ankle", "left_toe"),
    ("right_ankle", "right_toe"),
)

SOURCE_VIEWS = ("above", "under", "stitched")
