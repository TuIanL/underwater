from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from swim_pose.annotations import build_annotation_index, build_template
from swim_pose.constants import KEYPOINT_NAMES
from swim_pose.io import read_json, read_jsonl
from swim_pose.training.evaluate import evaluate_predictions_file
from swim_pose.training.inference import run_inference
from swim_pose.training.pseudolabels import generate_pseudolabel_file
from swim_pose.training.supervised import run_supervised_training
from swim_pose.training.yolo_pose import (
    PoseFramePrediction,
    PoseInstancePrediction,
    YOLOTrainingArtifact,
    export_yolo_pose_dataset,
)


class YOLOPoseDatasetExportTests(unittest.TestCase):
    def test_export_yolo_pose_dataset_writes_schema_and_18_keypoint_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            frame_root, annotation_root = _write_labeled_sequence(base, frame_count=2)
            index_path = build_annotation_index(annotation_root, base / "annotation_index.csv")

            bundle = export_yolo_pose_dataset(
                index_path=index_path,
                image_root=frame_root,
                output_dir=base / "yolo_dataset",
                val_index=index_path,
            )

            self.assertTrue(bundle.dataset_yaml.exists())
            self.assertTrue(bundle.schema_path.exists())
            schema = read_json(bundle.schema_path)
            self.assertIn("neck", schema["derived_keypoints"])
            self.assertIn("left_heel", schema["project_specific_keypoints"])

            label_files = sorted((bundle.root_dir / "labels" / "train").rglob("*.txt"))
            self.assertEqual(len(label_files), 2)
            fields = label_files[0].read_text(encoding="utf-8").strip().split()
            self.assertEqual(len(fields), 5 + len(KEYPOINT_NAMES) * 3)
            self.assertEqual(bundle.train_samples, 2)
            self.assertEqual(bundle.val_samples, 2)


class YOLOPoseTrainingAndInferenceTests(unittest.TestCase):
    def test_run_supervised_training_routes_to_yolo_and_writes_project_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            frame_root, annotation_root = _write_labeled_sequence(base, frame_count=2)
            index_path = build_annotation_index(annotation_root, base / "annotation_index.csv")
            config_path = _write_yolo_config(base, index_path, frame_root)
            runtime = _FakeTrainingRuntime()

            with patch("swim_pose.training.yolo_pose.create_yolo_runtime", return_value=runtime):
                checkpoint = run_supervised_training(config_path)

            state = torch.load(checkpoint, map_location="cpu")
            self.assertEqual(state["checkpoint_type"], "localization_yolo_pose")
            self.assertEqual(state["baseline_family"], "yolo_pose")
            self.assertIn("schema_adaptation", state)
            self.assertEqual(state["dataset_bundle"]["train_samples"], 2)
            self.assertEqual(len(runtime.train_calls), 1)

    def test_yolo_inference_adapts_upstream_17_keypoints_into_project_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            frame_root, annotation_root = _write_labeled_sequence(base, frame_count=2)
            index_path = build_annotation_index(annotation_root, base / "annotation_index.csv")
            config_path = _write_yolo_config(base, index_path, frame_root, postprocess_enabled=False)
            checkpoint_path = _write_yolo_checkpoint(base)
            runtime = _FakePredictRuntime(
                point_sequences=[
                    (_build_upstream_pose(frame_index=0), [0.9] * 17),
                    (_build_upstream_pose(frame_index=1), [0.9] * 17),
                ]
            )

            with patch("swim_pose.training.yolo_pose.create_yolo_runtime", return_value=runtime):
                predictions_path = run_inference(
                    config_path=config_path,
                    checkpoint_path=checkpoint_path,
                    index_path=index_path,
                    output_path=base / "predictions.jsonl",
                )

            predictions = read_jsonl(predictions_path)
            self.assertEqual(len(predictions), 2)
            first = predictions[0]["points"]
            self.assertIsNotNone(first["neck"]["x"])
            self.assertIsNone(first["left_heel"]["x"])
            self.assertIsNone(first["right_toe"]["x"])
            self.assertEqual(first["nose"]["visibility"], 2)
            self.assertEqual(predictions[0]["prediction_metadata"]["baseline_family"], "yolo_pose")
            self.assertNotIn("filtered_points", predictions[0])

    def test_temporal_postprocessing_exports_filtered_points_and_improves_midpoint_residual(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            frame_root, annotation_root = _write_labeled_sequence(base, frame_count=3)
            index_path = build_annotation_index(annotation_root, base / "annotation_index.csv")
            config_path = _write_yolo_config(base, index_path, frame_root, postprocess_enabled=True)
            checkpoint_path = _write_yolo_checkpoint(base)
            runtime = _FakePredictRuntime(
                point_sequences=[
                    (_build_project_pose(frame_index=0, jitter=0.0), [0.9] * len(KEYPOINT_NAMES)),
                    (_build_project_pose(frame_index=1, jitter=12.0), [0.9] * len(KEYPOINT_NAMES)),
                    (_build_project_pose(frame_index=2, jitter=0.0), [0.9] * len(KEYPOINT_NAMES)),
                ]
            )

            with patch("swim_pose.training.yolo_pose.create_yolo_runtime", return_value=runtime):
                predictions_path = run_inference(
                    config_path=config_path,
                    checkpoint_path=checkpoint_path,
                    index_path=index_path,
                    output_path=base / "predictions.jsonl",
                )

            predictions = read_jsonl(predictions_path)
            self.assertIn("filtered_points", predictions[1])
            raw_middle_x = predictions[1]["points"]["nose"]["x"]
            filtered_middle_x = predictions[1]["filtered_points"]["nose"]["x"]
            self.assertNotEqual(raw_middle_x, filtered_middle_x)

            filtered_sidecar = read_jsonl(base / "filtered_only.jsonl")
            self.assertEqual(len(filtered_sidecar), 3)
            self.assertEqual(filtered_sidecar[0]["prediction_variant"], "filtered")

            report_path = evaluate_predictions_file(
                predictions_path=predictions_path,
                annotations_path=index_path,
                output_path=base / "report.json",
            )
            report = read_json(report_path)
            self.assertIn("filtered_overall", report)
            self.assertLess(
                report["temporal_stability"]["filtered"]["mean_midpoint_residual"],
                report["temporal_stability"]["raw"]["mean_midpoint_residual"],
            )

            pseudolabels_path = generate_pseudolabel_file(
                predictions_path=predictions_path,
                output_path=base / "pseudolabels.jsonl",
                threshold=0.5,
                use_filtered=True,
            )
            pseudolabels = read_jsonl(pseudolabels_path)
            self.assertTrue(pseudolabels[0]["metadata"]["use_filtered"])
            self.assertEqual(
                pseudolabels[1]["points"]["nose"]["x"],
                predictions[1]["filtered_points"]["nose"]["x"],
            )


class _FakeTrainingRuntime:
    def __init__(self) -> None:
        self.train_calls: list[dict] = []

    def train(
        self,
        pretrained_model: str,
        data_config: Path,
        output_dir: Path,
        training_options: dict[str, object],
        adaptation: dict[str, object],
    ) -> YOLOTrainingArtifact:
        self.train_calls.append(
            {
                "pretrained_model": pretrained_model,
                "data_config": data_config,
                "output_dir": output_dir,
                "training_options": training_options,
                "adaptation": adaptation,
            }
        )
        weights_path = output_dir / "weights" / "best.pt"
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        weights_path.write_bytes(b"fake-yolo-weights")
        return YOLOTrainingArtifact(weights_path=weights_path, save_dir=output_dir)


class _FakePredictRuntime:
    def __init__(self, point_sequences: list[tuple[list[tuple[float, float]], list[float]]]) -> None:
        self.point_sequences = point_sequences
        self.offset = 0

    def predict(
        self,
        weights: str | Path,
        image_paths: list[Path],
        predict_options: dict[str, object],
    ) -> list[PoseFramePrediction]:
        predictions: list[PoseFramePrediction] = []
        for image_path in image_paths:
            keypoints, confidences = self.point_sequences[self.offset]
            self.offset += 1
            predictions.append(
                PoseFramePrediction(
                    image_path=str(image_path),
                    instances=[
                        PoseInstancePrediction(
                            keypoints=keypoints,
                            keypoint_confidences=confidences,
                            box_confidence=0.95,
                        )
                    ],
                )
            )
        return predictions


def _write_labeled_sequence(base: Path, frame_count: int) -> tuple[Path, Path]:
    frame_root = base / "frames"
    annotation_root = base / "annotations"
    for frame_index in range(frame_count):
        image_path = frame_root / "athlete01" / "session01" / "clipA" / "stitched" / f"frame_{frame_index:06d}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), color=(50 + frame_index * 10, 120, 180)).save(image_path)

        annotation = build_template()
        annotation["clip_id"] = "clipA"
        annotation["frame_index"] = frame_index
        annotation["source_view"] = "stitched"
        annotation["image_path"] = str(image_path)
        annotation["athlete_id"] = "athlete01"
        annotation["session_id"] = "session01"
        annotation["metadata"]["frame_status"] = "labeled"
        for keypoint_index, name in enumerate(KEYPOINT_NAMES):
            annotation["points"][name] = {
                "x": float(12 + frame_index * 2 + (keypoint_index % 4) * 3),
                "y": float(18 + frame_index * 2 + (keypoint_index % 5) * 2),
                "visibility": 2,
            }

        annotation_path = annotation_root / "athlete01" / "session01" / "clipA" / "stitched" / f"frame_{frame_index:06d}.json"
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        annotation_path.write_text(_dump_json(annotation), encoding="utf-8")
    return frame_root, annotation_root


def _write_yolo_config(base: Path, index_path: Path, frame_root: Path, postprocess_enabled: bool = False) -> Path:
    config_path = base / "yolo.toml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            [experiment]
            name = "yolo-smoke"
            output_dir = "{(base / 'artifacts').as_posix()}"
            seed = 7

            [dataset]
            train_index = "{index_path.as_posix()}"
            val_index = "{index_path.as_posix()}"
            annotation_index = "{index_path.as_posix()}"
            image_root = "{frame_root.as_posix()}"
            input_width = 64
            input_height = 64
            heatmap_width = 8
            heatmap_height = 8

            [model]
            family = "yolo_pose"

            [training]
            epochs = 1
            batch_size = 2
            learning_rate = 0.001
            weight_decay = 0.0001
            num_workers = 0
            device = "cpu"

            [yolo]
            pretrained_model = "fake-pose.pt"
            image_size = 64
            predict_batch_size = 2
            box_confidence = 0.25
            visibility_visible_threshold = 0.55
            visibility_inferable_threshold = 0.2
            adaptation_preset = "underwater_v1"
            dataset_dir = "{(base / 'yolo_dataset').as_posix()}"
            vendor_output_dir = "{(base / 'vendor').as_posix()}"

            [postprocess]
            enabled = {"true" if postprocess_enabled else "false"}
            method = "ema"
            alpha = 0.65
            min_alpha = 0.2
            confidence_floor = 0.1
            filtered_output = "{(base / 'filtered_only.jsonl').as_posix()}"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return config_path


def _write_yolo_checkpoint(base: Path) -> Path:
    checkpoint_path = base / "checkpoint.pt"
    vendor_weights = base / "vendor" / "weights" / "best.pt"
    vendor_weights.parent.mkdir(parents=True, exist_ok=True)
    vendor_weights.write_bytes(b"fake-vendor-weights")
    torch.save(
        {
            "checkpoint_type": "localization_yolo_pose",
            "vendor_checkpoint": vendor_weights.as_posix(),
        },
        checkpoint_path,
    )
    return checkpoint_path


def _build_upstream_pose(frame_index: int) -> list[tuple[float, float]]:
    return [
        (float(10 + frame_index * 2 + index), float(20 + frame_index + index))
        for index in range(17)
    ]


def _build_project_pose(frame_index: int, jitter: float) -> list[tuple[float, float]]:
    return [
        (
            float(12 + frame_index * 2 + (index % 4) * 3 + (jitter if index == 0 and frame_index == 1 else 0.0)),
            float(18 + frame_index * 2 + (index % 5) * 2 + (jitter if index == 0 and frame_index == 1 else 0.0)),
        )
        for index in range(len(KEYPOINT_NAMES))
    ]


def _dump_json(data: dict) -> str:
    import json

    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"
