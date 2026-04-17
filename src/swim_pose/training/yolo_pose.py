from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import shutil
from typing import Iterable

from PIL import Image
import torch

from ..constants import KEYPOINT_NAMES, KEYPOINT_SPECS
from ..io import ensure_parent, read_csv_rows, read_json, write_json, write_jsonl
from ..pathing import find_repo_root, resolve_repo_managed_path, serialize_workspace_path
from .baselines import YOLO_POSE_BASELINE
from .common import checkpoint_path, set_random_seed
from .config import experiment_output_dir, load_config
from .postprocess import apply_temporal_postprocessing, build_filtered_variant_rows


UPSTREAM_COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


@dataclass(frozen=True)
class YOLOTrainingArtifact:
    weights_path: Path
    save_dir: Path


@dataclass(frozen=True)
class PoseInstancePrediction:
    keypoints: list[tuple[float, float]]
    keypoint_confidences: list[float]
    box_confidence: float


@dataclass(frozen=True)
class PoseFramePrediction:
    image_path: str
    instances: list[PoseInstancePrediction]


@dataclass(frozen=True)
class YOLOPoseDatasetBundle:
    root_dir: Path
    dataset_yaml: Path
    schema_path: Path
    summary_path: Path
    train_samples: int
    val_samples: int


def create_yolo_runtime() -> "YOLOPoseRuntime":
    return YOLOPoseRuntime()


class YOLOPoseRuntime:
    def train(
        self,
        pretrained_model: str,
        data_config: Path,
        output_dir: Path,
        training_options: dict[str, object],
        adaptation: dict[str, object],
    ) -> YOLOTrainingArtifact:
        from ultralytics import YOLO

        destination = Path(output_dir).resolve()
        model = YOLO(pretrained_model)
        train_args = {
            "data": str(data_config),
            "epochs": int(training_options.get("epochs", 10)),
            "imgsz": int(training_options.get("image_size", 640)),
            "batch": int(training_options.get("batch_size", 8)),
            "lr0": float(training_options.get("learning_rate", 5e-4)),
            "weight_decay": float(training_options.get("weight_decay", 1e-4)),
            "device": training_options.get("device") or None,
            "seed": int(training_options.get("seed", 7)),
            "project": str(destination.parent),
            "name": destination.name,
            "exist_ok": True,
            "verbose": False,
        }
        train_args.update(adaptation.get("train_args", {}))
        filtered_args = {key: value for key, value in train_args.items() if value not in (None, "")}

        result = model.train(**filtered_args)
        save_dir = Path(getattr(result, "save_dir", destination)).resolve()
        trainer = getattr(model, "trainer", None)
        best_candidate = getattr(trainer, "best", None) if trainer is not None else None
        if best_candidate:
            weights_path = Path(best_candidate).resolve()
        else:
            weights_path = (save_dir / "weights" / "best.pt").resolve()
        return YOLOTrainingArtifact(weights_path=weights_path, save_dir=save_dir)

    def predict(
        self,
        weights: str | Path,
        image_paths: list[Path],
        predict_options: dict[str, object],
    ) -> list[PoseFramePrediction]:
        from ultralytics import YOLO

        model = YOLO(str(weights))
        predict_args = {
            "source": [str(path) for path in image_paths],
            "conf": float(predict_options.get("box_confidence", 0.25)),
            "imgsz": int(predict_options.get("image_size", 640)),
            "device": predict_options.get("device") or None,
            "verbose": False,
        }
        filtered_args = {key: value for key, value in predict_args.items() if value not in (None, "")}
        results = model.predict(**filtered_args)
        return [self._convert_result(path, result) for path, result in zip(image_paths, results)]

    def _convert_result(self, image_path: Path, result: object) -> PoseFramePrediction:
        keypoints = getattr(result, "keypoints", None)
        boxes = getattr(result, "boxes", None)
        instances: list[PoseInstancePrediction] = []
        if keypoints is None or getattr(keypoints, "xy", None) is None:
            return PoseFramePrediction(image_path=str(image_path), instances=instances)

        xy_tensor = keypoints.xy.cpu()
        conf_tensor = keypoints.conf.cpu() if getattr(keypoints, "conf", None) is not None else None
        box_confidences = boxes.conf.cpu().tolist() if boxes is not None and getattr(boxes, "conf", None) is not None else []
        for index in range(xy_tensor.shape[0]):
            xy_values = xy_tensor[index].tolist()
            point_confidences = (
                conf_tensor[index].tolist() if conf_tensor is not None else [box_confidences[index] if index < len(box_confidences) else 0.0] * len(xy_values)
            )
            instances.append(
                PoseInstancePrediction(
                    keypoints=[(float(x), float(y)) for x, y in xy_values],
                    keypoint_confidences=[float(value) for value in point_confidences],
                    box_confidence=float(box_confidences[index]) if index < len(box_confidences) else 0.0,
                )
            )
        return PoseFramePrediction(image_path=str(image_path), instances=instances)


def run_yolo_pose_training(config_path: str | Path) -> Path:
    config = load_config(config_path)
    set_random_seed(int(config["experiment"].get("seed", 7)))
    output_dir = experiment_output_dir(config)
    dataset_config = config["dataset"]
    training_config = config["training"]
    yolo_config = config.get("yolo", {})

    train_index = dataset_config.get("train_index") or dataset_config.get("annotation_index")
    if not train_index:
        raise ValueError("dataset.train_index or dataset.annotation_index must be configured")

    dataset_dir = Path(yolo_config.get("dataset_dir") or (output_dir / "yolo_dataset"))
    bundle = export_yolo_pose_dataset(
        index_path=train_index,
        image_root=dataset_config["image_root"],
        output_dir=dataset_dir,
        val_index=dataset_config.get("val_index", ""),
    )

    adaptation = build_underwater_adaptation_config(config)
    runtime = create_yolo_runtime()
    artifact = runtime.train(
        pretrained_model=str(yolo_config.get("pretrained_model", "yolov8s-pose.pt")),
        data_config=bundle.dataset_yaml,
        output_dir=Path(yolo_config.get("vendor_output_dir") or (output_dir / "vendor")),
        training_options={
            "epochs": training_config.get("epochs", 10),
            "image_size": yolo_config.get("image_size", max(dataset_config["input_width"], dataset_config["input_height"])),
            "batch_size": training_config.get("batch_size", 8),
            "learning_rate": training_config.get("learning_rate", 5e-4),
            "weight_decay": training_config.get("weight_decay", 1e-4),
            "device": training_config.get("device"),
            "seed": config["experiment"].get("seed", 7),
        },
        adaptation=adaptation,
    )

    checkpoint = checkpoint_path(output_dir)
    torch.save(
        {
            "checkpoint_type": "localization_yolo_pose",
            "training_mode": "supervised",
            "prediction_contract": "frame_keyed",
            "baseline_family": YOLO_POSE_BASELINE,
            "vendor_checkpoint": serialize_workspace_path(artifact.weights_path),
            "vendor_run_dir": serialize_workspace_path(artifact.save_dir),
            "schema_adaptation": build_schema_adaptation_metadata(),
            "underwater_domain_adaptation": adaptation,
            "dataset_bundle": {
                "dataset_yaml": serialize_workspace_path(bundle.dataset_yaml),
                "schema_path": serialize_workspace_path(bundle.schema_path),
                "summary_path": serialize_workspace_path(bundle.summary_path),
                "train_samples": bundle.train_samples,
                "val_samples": bundle.val_samples,
            },
            "config": config,
        },
        checkpoint,
    )
    return checkpoint


def run_yolo_pose_inference(
    config_path: str | Path,
    checkpoint_path: str | Path,
    index_path: str | Path,
    output_path: str | Path,
    labeled: bool = True,
) -> Path:
    config = load_config(config_path)
    state = torch.load(checkpoint_path, map_location="cpu")
    if state.get("checkpoint_type") != "localization_yolo_pose":
        raise ValueError("YOLO pose inference expects a localization_yolo_pose checkpoint.")

    yolo_config = config.get("yolo", {})
    predict_rows = _load_prediction_rows(
        index_path=index_path,
        image_root=config["dataset"]["image_root"],
        labeled=labeled,
    )
    runtime = create_yolo_runtime()
    weights_reference = state.get("vendor_checkpoint")
    weights = (
        resolve_repo_managed_path(weights_reference)
        if isinstance(weights_reference, str) and weights_reference.endswith(".pt")
        else weights_reference
    )
    batch_size = max(int(yolo_config.get("predict_batch_size", 8)), 1)
    outputs: list[dict] = []
    for batch in _chunked(predict_rows, batch_size):
        frame_predictions = runtime.predict(
            weights=weights,
            image_paths=[Path(row["_resolved_image_path"]) for row in batch],
            predict_options={
                "box_confidence": yolo_config.get("box_confidence", 0.25),
                "image_size": yolo_config.get(
                    "image_size",
                    max(config["dataset"]["input_width"], config["dataset"]["input_height"]),
                ),
                "device": config["training"].get("device"),
            },
        )
        for row, frame_prediction in zip(batch, frame_predictions):
            outputs.append(
                _format_project_prediction(
                    row=row,
                    frame_prediction=frame_prediction,
                    visible_threshold=float(yolo_config.get("visibility_visible_threshold", 0.55)),
                    inferable_threshold=float(yolo_config.get("visibility_inferable_threshold", 0.2)),
                )
            )

    outputs = apply_temporal_postprocessing(outputs, config)
    write_jsonl(output_path, outputs)
    filtered_output = str(config.get("postprocess", {}).get("filtered_output", "")).strip()
    if filtered_output:
        write_jsonl(filtered_output, build_filtered_variant_rows(outputs))
    return Path(output_path)


def export_yolo_pose_dataset(
    index_path: str | Path,
    image_root: str | Path,
    output_dir: str | Path,
    val_index: str | Path | None = None,
) -> YOLOPoseDatasetBundle:
    rows = read_csv_rows(index_path)
    if not rows:
        raise ValueError(f"No labeled rows found in index: {index_path}")

    val_rows = read_csv_rows(val_index) if val_index and Path(val_index).exists() else []
    if not val_rows:
        val_rows = rows

    root_dir = Path(output_dir).resolve()
    for relative in ("images/train", "images/val", "labels/train", "labels/val"):
        destination = root_dir / relative
        if destination.exists():
            shutil.rmtree(destination)

    repo_root = find_repo_root(index_path) or Path.cwd()
    train_samples = _export_split_rows(
        rows=rows,
        split_name="train",
        image_root=image_root,
        output_dir=root_dir,
        repo_root=repo_root,
    )
    val_samples = _export_split_rows(
        rows=val_rows,
        split_name="val",
        image_root=image_root,
        output_dir=root_dir,
        repo_root=repo_root,
    )

    dataset_yaml = ensure_parent(root_dir / "dataset.yaml")
    dataset_yaml.write_text(_render_dataset_yaml(root_dir), encoding="utf-8")
    schema_path = root_dir / "schema.json"
    write_json(schema_path, build_schema_adaptation_metadata())
    summary_path = root_dir / "export_summary.json"
    write_json(
        summary_path,
        {
            "train_samples": train_samples,
            "val_samples": val_samples,
            "project_keypoints": list(KEYPOINT_NAMES),
            "flip_idx": _project_flip_indices(),
        },
    )
    return YOLOPoseDatasetBundle(
        root_dir=root_dir,
        dataset_yaml=dataset_yaml,
        schema_path=schema_path,
        summary_path=summary_path,
        train_samples=train_samples,
        val_samples=val_samples,
    )


def build_schema_adaptation_metadata() -> dict[str, object]:
    return {
        "project_keypoints": list(KEYPOINT_NAMES),
        "upstream_keypoints": list(UPSTREAM_COCO_KEYPOINT_NAMES),
        "shared_keypoints": {
            "nose": "nose",
            "left_shoulder": "left_shoulder",
            "right_shoulder": "right_shoulder",
            "left_elbow": "left_elbow",
            "right_elbow": "right_elbow",
            "left_wrist": "left_wrist",
            "right_wrist": "right_wrist",
            "left_hip": "left_hip",
            "right_hip": "right_hip",
            "left_knee": "left_knee",
            "right_knee": "right_knee",
            "left_ankle": "left_ankle",
            "right_ankle": "right_ankle",
        },
        "derived_keypoints": {
            "neck": "midpoint(left_shoulder,right_shoulder)",
        },
        "project_specific_keypoints": {
            "left_heel": "directly supervised in the project dataset",
            "right_heel": "directly supervised in the project dataset",
            "left_toe": "directly supervised in the project dataset",
            "right_toe": "directly supervised in the project dataset",
        },
        "flip_idx": _project_flip_indices(),
        "keypoint_groups": {spec.name: spec.group for spec in KEYPOINT_SPECS},
    }


def build_underwater_adaptation_config(config: dict) -> dict[str, object]:
    yolo_config = config.get("yolo", {})
    preset_name = str(yolo_config.get("adaptation_preset", "underwater_v1")).strip().lower()
    presets = {
        "plain_pose": {
            "focus": [],
            "train_args": {
                "hsv_h": 0.015,
                "hsv_s": 0.3,
                "hsv_v": 0.2,
                "translate": 0.05,
                "scale": 0.15,
                "perspective": 0.0,
                "fliplr": 0.0,
                "mosaic": 0.0,
                "mixup": 0.0,
            },
        },
        "underwater_v1": {
            "focus": ["splash", "blur", "seam_artifacts", "self_occlusion"],
            "train_args": {
                "hsv_h": 0.015,
                "hsv_s": 0.5,
                "hsv_v": 0.3,
                "translate": 0.08,
                "scale": 0.25,
                "perspective": 0.0005,
                "fliplr": 0.0,
                "mosaic": 0.0,
                "mixup": 0.0,
            },
        },
    }
    if preset_name not in presets:
        supported = ", ".join(sorted(presets))
        raise ValueError(f"Unsupported YOLO adaptation preset '{preset_name}'. Expected one of: {supported}")

    preset = dict(presets[preset_name])
    preset["name"] = preset_name
    return preset


def _export_split_rows(
    rows: list[dict[str, str]],
    split_name: str,
    image_root: str | Path,
    output_dir: Path,
    repo_root: Path,
) -> int:
    image_dir = output_dir / "images" / split_name
    label_dir = output_dir / "labels" / split_name
    sample_count = 0
    for row in rows:
        annotation_path = resolve_repo_managed_path(row["annotation_path"], repo_root)
        annotation = read_json(annotation_path)
        resolved_image_path = _resolve_image_path(annotation.get("image_path", row.get("image_path", "")), image_root)
        relative_path = _relative_export_path(row=row, image_path=resolved_image_path)
        destination_image = image_dir / relative_path
        destination_label = label_dir / relative_path.with_suffix(".txt")
        _mirror_image(resolved_image_path, destination_image)
        label_line = _build_label_line(annotation=annotation, image_path=resolved_image_path)
        ensure_parent(destination_label).write_text(label_line + "\n", encoding="utf-8")
        sample_count += 1
    return sample_count


def _build_label_line(annotation: dict, image_path: Path) -> str:
    with Image.open(image_path) as image:
        width, height = image.size

    visible_points: list[tuple[float, float]] = []
    keypoint_values: list[str] = []
    for name in KEYPOINT_NAMES:
        point = annotation["points"][name]
        visibility = int(point.get("visibility", 0))
        x = point.get("x")
        y = point.get("y")
        if visibility > 0 and x is not None and y is not None:
            normalized_x = min(max(float(x) / max(width, 1), 0.0), 1.0)
            normalized_y = min(max(float(y) / max(height, 1), 0.0), 1.0)
            visible_points.append((float(x), float(y)))
            keypoint_values.extend([f"{normalized_x:.6f}", f"{normalized_y:.6f}", str(visibility)])
        else:
            keypoint_values.extend(["0.000000", "0.000000", "0"])

    if not visible_points:
        raise ValueError(f"Annotation '{annotation.get('clip_id', '')}' has no visible keypoints for YOLO export.")

    xs = [point[0] for point in visible_points]
    ys = [point[1] for point in visible_points]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    padding_x = max((max_x - min_x) * 0.05, width * 0.02, 1.0)
    padding_y = max((max_y - min_y) * 0.05, height * 0.02, 1.0)
    x1 = max(min_x - padding_x, 0.0)
    y1 = max(min_y - padding_y, 0.0)
    x2 = min(max_x + padding_x, float(width))
    y2 = min(max_y + padding_y, float(height))

    center_x = ((x1 + x2) / 2.0) / max(width, 1)
    center_y = ((y1 + y2) / 2.0) / max(height, 1)
    box_width = (x2 - x1) / max(width, 1)
    box_height = (y2 - y1) / max(height, 1)
    fields = [
        "0",
        f"{center_x:.6f}",
        f"{center_y:.6f}",
        f"{box_width:.6f}",
        f"{box_height:.6f}",
        *keypoint_values,
    ]
    return " ".join(fields)


def _render_dataset_yaml(root_dir: Path) -> str:
    flip_idx = ", ".join(str(value) for value in _project_flip_indices())
    return "\n".join(
        [
            f"path: {root_dir.as_posix()}",
            "train: images/train",
            "val: images/val",
            "kpt_shape: [18, 3]",
            f"flip_idx: [{flip_idx}]",
            "names:",
            "  0: swimmer",
            "",
        ]
    )


def _relative_export_path(row: dict[str, str], image_path: Path) -> Path:
    athlete_id = _safe_path_token(row.get("athlete_id") or "unknown_athlete")
    session_id = _safe_path_token(row.get("session_id") or "unknown_session")
    clip_id = _safe_path_token(row.get("clip_id") or image_path.stem)
    source_view = _safe_path_token(row.get("source_view") or "stitched")
    return Path(athlete_id) / session_id / clip_id / source_view / image_path.name


def _safe_path_token(value: str) -> str:
    token = str(value).strip().replace("\\", "_").replace("/", "_")
    return token or "unknown"


def _mirror_image(source: Path, destination: Path) -> None:
    destination_path = ensure_parent(destination)
    if destination_path.exists():
        return
    try:
        destination_path.symlink_to(source)
    except OSError:
        shutil.copy2(source, destination_path)


def _load_prediction_rows(index_path: str | Path, image_root: str | Path, labeled: bool) -> list[dict]:
    rows = read_csv_rows(index_path)
    repo_root = find_repo_root(index_path) or Path.cwd()
    loaded_rows: list[dict] = []
    for row in rows:
        annotation = read_json(resolve_repo_managed_path(row["annotation_path"], repo_root)) if labeled else {}
        image_path = annotation.get("image_path") or row.get("image_path", "")
        resolved_image_path = _resolve_image_path(image_path, image_root)
        with Image.open(resolved_image_path) as image:
            original_width, original_height = image.size
        loaded_rows.append(
            {
                "annotation_path": row.get("annotation_path", ""),
                "clip_id": row.get("clip_id", annotation.get("clip_id", "")),
                "athlete_id": row.get("athlete_id", annotation.get("athlete_id", "")),
                "session_id": row.get("session_id", annotation.get("session_id", "")),
                "frame_index": int(row.get("frame_index", annotation.get("frame_index", 0))),
                "source_view": row.get("source_view", annotation.get("source_view", "")),
                "image_path": image_path,
                "original_width": original_width,
                "original_height": original_height,
                "_resolved_image_path": str(resolved_image_path),
            }
        )
    return loaded_rows


def _resolve_image_path(image_path: str, image_root: str | Path) -> Path:
    candidate = Path(image_path)
    if candidate.is_absolute():
        return candidate.resolve()
    existing = candidate.resolve() if candidate.exists() else None
    if existing is not None:
        return existing
    return (Path(image_root) / candidate).resolve()


def _format_project_prediction(
    row: dict,
    frame_prediction: PoseFramePrediction,
    visible_threshold: float,
    inferable_threshold: float,
) -> dict:
    instance = _select_primary_instance(frame_prediction.instances)
    if instance is None:
        points = {
            name: {
                "x": None,
                "y": None,
                "confidence": 0.0,
                "visibility": 0,
            }
            for name in KEYPOINT_NAMES
        }
        box_confidence = 0.0
    else:
        points = _adapt_points_to_project_schema(
            instance=instance,
            visible_threshold=visible_threshold,
            inferable_threshold=inferable_threshold,
        )
        box_confidence = instance.box_confidence
    return {
        "annotation_path": row.get("annotation_path", ""),
        "clip_id": row.get("clip_id", ""),
        "athlete_id": row.get("athlete_id", ""),
        "session_id": row.get("session_id", ""),
        "frame_index": int(row.get("frame_index", 0)),
        "source_view": row.get("source_view", ""),
        "image_path": row.get("image_path", ""),
        "points": points,
        "prediction_metadata": {
            "baseline_family": YOLO_POSE_BASELINE,
            "box_confidence": box_confidence,
        },
    }


def _adapt_points_to_project_schema(
    instance: PoseInstancePrediction,
    visible_threshold: float,
    inferable_threshold: float,
) -> dict[str, dict]:
    point_count = len(instance.keypoints)
    if point_count == len(KEYPOINT_NAMES):
        raw_points = {
            name: _project_point_from_runtime(
                x=instance.keypoints[index][0],
                y=instance.keypoints[index][1],
                confidence=_confidence_at(instance.keypoint_confidences, index),
                visible_threshold=visible_threshold,
                inferable_threshold=inferable_threshold,
            )
            for index, name in enumerate(KEYPOINT_NAMES)
        }
        return raw_points
    if point_count == len(UPSTREAM_COCO_KEYPOINT_NAMES):
        upstream_points = {
            name: (
                instance.keypoints[index][0],
                instance.keypoints[index][1],
                _confidence_at(instance.keypoint_confidences, index),
            )
            for index, name in enumerate(UPSTREAM_COCO_KEYPOINT_NAMES)
        }
        project_points = {
            "nose": _project_point_from_tuple(upstream_points["nose"], visible_threshold, inferable_threshold),
            "left_shoulder": _project_point_from_tuple(upstream_points["left_shoulder"], visible_threshold, inferable_threshold),
            "right_shoulder": _project_point_from_tuple(upstream_points["right_shoulder"], visible_threshold, inferable_threshold),
            "left_elbow": _project_point_from_tuple(upstream_points["left_elbow"], visible_threshold, inferable_threshold),
            "right_elbow": _project_point_from_tuple(upstream_points["right_elbow"], visible_threshold, inferable_threshold),
            "left_wrist": _project_point_from_tuple(upstream_points["left_wrist"], visible_threshold, inferable_threshold),
            "right_wrist": _project_point_from_tuple(upstream_points["right_wrist"], visible_threshold, inferable_threshold),
            "left_hip": _project_point_from_tuple(upstream_points["left_hip"], visible_threshold, inferable_threshold),
            "right_hip": _project_point_from_tuple(upstream_points["right_hip"], visible_threshold, inferable_threshold),
            "left_knee": _project_point_from_tuple(upstream_points["left_knee"], visible_threshold, inferable_threshold),
            "right_knee": _project_point_from_tuple(upstream_points["right_knee"], visible_threshold, inferable_threshold),
            "left_ankle": _project_point_from_tuple(upstream_points["left_ankle"], visible_threshold, inferable_threshold),
            "right_ankle": _project_point_from_tuple(upstream_points["right_ankle"], visible_threshold, inferable_threshold),
        }
        project_points["neck"] = _derive_neck(
            left_shoulder=upstream_points["left_shoulder"],
            right_shoulder=upstream_points["right_shoulder"],
            visible_threshold=visible_threshold,
            inferable_threshold=inferable_threshold,
        )
        for missing_name in ("left_heel", "right_heel", "left_toe", "right_toe"):
            project_points[missing_name] = {
                "x": None,
                "y": None,
                "confidence": 0.0,
                "visibility": 0,
            }
        return {name: project_points[name] for name in KEYPOINT_NAMES}
    raise ValueError(
        f"Unsupported runtime keypoint count {point_count}. "
        f"Expected {len(KEYPOINT_NAMES)} project keypoints or {len(UPSTREAM_COCO_KEYPOINT_NAMES)} upstream keypoints."
    )


def _derive_neck(
    left_shoulder: tuple[float, float, float],
    right_shoulder: tuple[float, float, float],
    visible_threshold: float,
    inferable_threshold: float,
) -> dict:
    left_confidence = float(left_shoulder[2])
    right_confidence = float(right_shoulder[2])
    if left_confidence <= 0.0 or right_confidence <= 0.0:
        return {"x": None, "y": None, "confidence": 0.0, "visibility": 0}
    x = (float(left_shoulder[0]) + float(right_shoulder[0])) / 2.0
    y = (float(left_shoulder[1]) + float(right_shoulder[1])) / 2.0
    confidence = min(left_confidence, right_confidence)
    return _project_point_from_runtime(
        x=x,
        y=y,
        confidence=confidence,
        visible_threshold=visible_threshold,
        inferable_threshold=inferable_threshold,
    )


def _project_point_from_tuple(
    point: tuple[float, float, float],
    visible_threshold: float,
    inferable_threshold: float,
) -> dict:
    x, y, confidence = point
    return _project_point_from_runtime(
        x=x,
        y=y,
        confidence=confidence,
        visible_threshold=visible_threshold,
        inferable_threshold=inferable_threshold,
    )


def _project_point_from_runtime(
    x: float,
    y: float,
    confidence: float,
    visible_threshold: float,
    inferable_threshold: float,
) -> dict:
    if confidence < inferable_threshold or math.isnan(float(confidence)):
        return {
            "x": None,
            "y": None,
            "confidence": float(confidence),
            "visibility": 0,
        }
    visibility = 2 if confidence >= visible_threshold else 1
    return {
        "x": float(x),
        "y": float(y),
        "confidence": float(confidence),
        "visibility": visibility,
    }


def _select_primary_instance(instances: Iterable[PoseInstancePrediction]) -> PoseInstancePrediction | None:
    instance_list = list(instances)
    if not instance_list:
        return None
    return max(
        instance_list,
        key=lambda instance: (
            float(instance.box_confidence),
            sum(float(value) for value in instance.keypoint_confidences) / max(len(instance.keypoint_confidences), 1),
        ),
    )


def _confidence_at(values: list[float], index: int) -> float:
    return float(values[index]) if index < len(values) else 0.0


def _project_flip_indices() -> list[int]:
    mirrored_names = {
        "left_shoulder": "right_shoulder",
        "right_shoulder": "left_shoulder",
        "left_elbow": "right_elbow",
        "right_elbow": "left_elbow",
        "left_wrist": "right_wrist",
        "right_wrist": "left_wrist",
        "left_hip": "right_hip",
        "right_hip": "left_hip",
        "left_knee": "right_knee",
        "right_knee": "left_knee",
        "left_ankle": "right_ankle",
        "right_ankle": "left_ankle",
        "left_heel": "right_heel",
        "right_heel": "left_heel",
        "left_toe": "right_toe",
        "right_toe": "left_toe",
    }
    return [KEYPOINT_NAMES.index(mirrored_names.get(name, name)) for name in KEYPOINT_NAMES]


def _chunked(rows: list[dict], batch_size: int) -> Iterable[list[dict]]:
    for index in range(0, len(rows), batch_size):
        yield rows[index : index + batch_size]
