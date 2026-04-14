from __future__ import annotations

import json
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from swim_pose.annotations import build_annotation_index, build_template
from swim_pose.training.dataset import TemporalPoseDataset
from swim_pose.training.model import build_model
from swim_pose.training.supervised import run_supervised_training


class LocalizationBridgeDatasetTests(unittest.TestCase):
    def test_temporal_pose_dataset_returns_neighbor_clip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            frame_root, annotation_root = _write_labeled_sequence(base, frame_count=5, clip_id="clipA")
            index_path = build_annotation_index(annotation_root, base / "annotation_index.csv")

            dataset = TemporalPoseDataset(
                index_path=index_path,
                image_root=frame_root,
                input_size=(32, 32),
                heatmap_size=(8, 8),
                bridge_input_size=(16, 16),
                bridge_clip_length=3,
                bridge_frame_stride=1,
            )

            sample = dataset[2]

            self.assertEqual(sample["bridge_clip"].shape, (3, 3, 16, 16))
            self.assertEqual(sample["bridge_frame_indices"].tolist(), [1, 2, 3])
            frame_means = sample["bridge_clip"].mean(dim=(0, 2, 3)).tolist()
            self.assertLess(frame_means[0], frame_means[1])
            self.assertLess(frame_means[1], frame_means[2])


class LocalizationBridgeTrainingTests(unittest.TestCase):
    def test_run_supervised_training_writes_bridge_checkpoint_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            frame_root, annotation_root = _write_labeled_sequence(base, frame_count=3, clip_id="clipB")
            index_path = build_annotation_index(annotation_root, base / "annotation_index.csv")
            teacher_checkpoint = base / "teacher.pt"
            teacher_checkpoint.write_bytes(b"bridge")
            config_path = base / "supervised_bridge.toml"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    [experiment]
                    name = "supervised-bridge-smoke"
                    output_dir = "{(base / 'artifacts').as_posix()}"
                    seed = 7

                    [dataset]
                    train_index = "{index_path.as_posix()}"
                    annotation_index = "{index_path.as_posix()}"
                    image_root = "{frame_root.as_posix()}"
                    input_width = 32
                    input_height = 32
                    heatmap_width = 8
                    heatmap_height = 8

                    [model]
                    backbone = "resnet18"
                    pretrained_backbone = false
                    pretrained_checkpoint = ""

                    [training]
                    epochs = 1
                    batch_size = 2
                    learning_rate = 0.0005
                    weight_decay = 0.0001
                    num_workers = 0
                    visibility_loss_weight = 0.2
                    device = "cpu"

                    [bridge]
                    enabled = true
                    teacher_checkpoint = "{teacher_checkpoint.as_posix()}"
                    input_width = 16
                    input_height = 16
                    clip_length = 3
                    frame_stride = 1
                    distillation_weight = 0.1
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            with patch("swim_pose.training.supervised.load_bridge_teacher", return_value=(_FakeTeacher(), {"backbone": "fake_video_teacher"})):
                checkpoint = run_supervised_training(config_path)

            metrics_path = base / "artifacts" / "train_metrics.json"
            state = torch.load(checkpoint, map_location="cpu")
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))

            self.assertEqual(state["checkpoint_type"], "localization_bridge")
            self.assertEqual(state["training_mode"], "supervised")
            self.assertEqual(state["bridge_type"], "video_feature_distillation")
            self.assertEqual(state["bridge_teacher_backbone"], "fake_video_teacher")
            self.assertIn("bridge_projector", state)
            self.assertEqual(len(payload), 1)
            self.assertIn("bridge_loss", payload[0])

    def test_run_supervised_training_without_bridge_writes_localization_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            frame_root, annotation_root = _write_labeled_sequence(base, frame_count=2, clip_id="clipC")
            index_path = build_annotation_index(annotation_root, base / "annotation_index.csv")
            config_path = base / "supervised.toml"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    [experiment]
                    name = "supervised-smoke"
                    output_dir = "{(base / 'artifacts').as_posix()}"
                    seed = 7

                    [dataset]
                    train_index = "{index_path.as_posix()}"
                    annotation_index = "{index_path.as_posix()}"
                    image_root = "{frame_root.as_posix()}"
                    input_width = 32
                    input_height = 32
                    heatmap_width = 8
                    heatmap_height = 8

                    [model]
                    backbone = "resnet18"
                    pretrained_backbone = false
                    pretrained_checkpoint = ""

                    [training]
                    epochs = 1
                    batch_size = 2
                    learning_rate = 0.0005
                    weight_decay = 0.0001
                    num_workers = 0
                    visibility_loss_weight = 0.2
                    device = "cpu"
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            checkpoint = run_supervised_training(config_path)
            state = torch.load(checkpoint, map_location="cpu")

            self.assertEqual(state["checkpoint_type"], "localization")
            self.assertEqual(state["training_mode"], "supervised")
            self.assertNotIn("bridge_projector", state)

    def test_build_model_accepts_localization_bridge_checkpoint(self) -> None:
        config = {"model": {"backbone": "resnet18", "pretrained_backbone": False, "pretrained_checkpoint": ""}}
        model = build_model(config, num_keypoints=18)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "bridge.pt"
            torch.save(
                {
                    "checkpoint_type": "localization_bridge",
                    "training_mode": "supervised",
                    "model": model.state_dict(),
                    "bridge_projector": {"layers.0.weight": torch.randn(4, 4)},
                },
                checkpoint_path,
            )

            reloaded = build_model(
                {
                    "model": {
                        "backbone": "resnet18",
                        "pretrained_backbone": False,
                        "pretrained_checkpoint": str(checkpoint_path),
                    }
                },
                num_keypoints=18,
            )

            self.assertIsNotNone(reloaded)


class _FakeTeacher(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_dim = 6

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        pooled = clips.mean(dim=(2, 3, 4))
        padding = torch.zeros((clips.shape[0], self.feature_dim - pooled.shape[1]), device=clips.device)
        return torch.cat([pooled, padding], dim=1)


def _write_labeled_sequence(base: Path, frame_count: int, clip_id: str) -> tuple[Path, Path]:
    frame_root = base / "frames"
    annotation_root = base / "annotations"
    athlete_id = "athlete01"
    session_id = "session01"
    source_view = "stitched"
    for frame_index in range(frame_count):
        relative_image_path = (
            Path(athlete_id)
            / session_id
            / clip_id
            / source_view
            / f"{clip_id}_{source_view}_{frame_index:06d}.jpg"
        )
        image_path = frame_root / relative_image_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 12), color=(frame_index * 40, frame_index * 40, frame_index * 40)).save(image_path)

        annotation = build_template()
        annotation["clip_id"] = clip_id
        annotation["athlete_id"] = athlete_id
        annotation["session_id"] = session_id
        annotation["source_view"] = source_view
        annotation["frame_index"] = frame_index
        annotation["image_path"] = relative_image_path.as_posix()
        annotation["metadata"]["frame_status"] = "labeled"
        annotation_path = (
            annotation_root
            / athlete_id
            / session_id
            / clip_id
            / source_view
            / f"{clip_id}_{source_view}_{frame_index:06d}.json"
        )
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        annotation_path.write_text(json.dumps(annotation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return frame_root, annotation_root


if __name__ == "__main__":
    unittest.main()
