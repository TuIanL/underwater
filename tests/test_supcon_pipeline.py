from __future__ import annotations

import json
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from swim_pose.io import read_csv_rows
from swim_pose.manifest import (
    build_supcon_video_index,
    inspect_supcon_video,
    parse_supcon_video_stem,
)
from swim_pose.training import dataset as dataset_module
from swim_pose.training.dataset import SupConVideoDataset
from swim_pose.training.model import build_model, build_supcon_model
from swim_pose.training.supcon import run_supcon_training


class SupConIngestionTests(unittest.TestCase):
    def test_parse_supcon_video_stem_supports_canonical_tokens(self) -> None:
        self.assertEqual(parse_supcon_video_stem("蛙"), ("蛙", "", "valid", ""))
        self.assertEqual(parse_supcon_video_stem("仰_02"), ("仰", "02", "valid", ""))
        self.assertEqual(parse_supcon_video_stem("蝶-test"), ("蝶", "test", "valid", ""))

    def test_inspect_supcon_video_classifies_invalid_and_mixed_inputs(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            video_root = Path(tmp) / "videos"

            mixed_path = video_root / "athlete01" / "session01" / "四式合集.avi"
            mixed_path.parent.mkdir(parents=True, exist_ok=True)
            mixed_path.write_bytes(b"")
            mixed_entry = inspect_supcon_video(video_root, mixed_path, repo_root=repo_root)
            self.assertEqual(mixed_entry.validation_status, "mixed_stroke")

            invalid_layout_path = video_root / "athlete01" / "session01" / "nested" / "蛙_01.avi"
            invalid_layout_path.parent.mkdir(parents=True, exist_ok=True)
            invalid_layout_path.write_bytes(b"")
            invalid_layout_entry = inspect_supcon_video(video_root, invalid_layout_path, repo_root=repo_root)
            self.assertEqual(invalid_layout_entry.validation_status, "invalid_layout")

    def test_build_supcon_video_index_preserves_validation_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            video_root = base / "videos"
            _write_test_video(video_root / "athlete01" / "session01" / "蛙_01.avi")
            mixed_path = video_root / "athlete01" / "session01" / "四式合集.avi"
            mixed_path.write_bytes(b"")

            destination, summary = build_supcon_video_index(video_root, base / "supcon.csv")
            rows = read_csv_rows(destination)

            self.assertEqual(summary["rows"], 2)
            self.assertEqual(summary["valid"], 1)
            self.assertEqual(summary["mixed_stroke"], 1)
            by_stem = {Path(row["video_path"]).stem: row for row in rows}
            self.assertEqual(by_stem["蛙_01"]["validation_status"], "valid")
            self.assertEqual(by_stem["四式合集"]["validation_status"], "mixed_stroke")


class SupConTrainingSmokeTests(unittest.TestCase):
    def test_supcon_dataset_reuses_one_sampled_clip_for_both_views(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            video_root = base / "videos"
            _write_test_video(video_root / "athlete01" / "session01" / "蛙_01.avi", frame_count=8)
            index_path, summary = build_supcon_video_index(video_root, base / "supcon.csv")
            self.assertEqual(summary["valid"], 1)

            dataset = SupConVideoDataset(
                index_path=index_path,
                input_size=(32, 32),
                clip_length=4,
                frame_stride=1,
                temporal_jitter=0,
                crop_scale_range=(1.0, 1.0),
                color_jitter_strength=0.0,
                grayscale_prob=0.0,
                blur_prob=0.0,
                blur_kernel_size=3,
            )
            raw_clip = torch.arange(3 * 4 * 32 * 32, dtype=torch.float32).reshape(3, 4, 32, 32) / 255.0
            load_calls: list[list[int]] = []
            augment_call_count = 0

            def fake_load_video_clip(path: Path, frame_indices: list[int], input_size: tuple[int, int]) -> torch.Tensor:
                self.assertEqual(input_size, (32, 32))
                self.assertTrue(path.exists())
                load_calls.append(list(frame_indices))
                return raw_clip.clone()

            def fake_augment_video_clip(clip: torch.Tensor, **_: object) -> torch.Tensor:
                nonlocal augment_call_count
                augment_call_count += 1
                return clip + augment_call_count

            with (
                patch.object(dataset_module, "_load_video_clip", side_effect=fake_load_video_clip),
                patch.object(dataset_module, "_augment_video_clip", side_effect=fake_augment_video_clip),
            ):
                sample = dataset[0]

            self.assertEqual(len(load_calls), 1)
            self.assertEqual(sample["clip_frame_indices"].tolist(), load_calls[0])
            self.assertFalse(torch.equal(sample["view_1"], sample["view_2"]))
            self.assertTrue(
                torch.allclose(sample["view_1"] - sample["view_2"], torch.full_like(raw_clip, -1.0), atol=1e-5)
            )

    def test_build_supcon_model_uses_video_backbone(self) -> None:
        config = {
            "model": {
                "backbone": "r2plus1d_18",
                "pretrained_backbone": False,
                "pretrained_checkpoint": "",
                "projection_hidden_dim": 64,
                "projection_dim": 16,
            }
        }

        model = build_supcon_model(config)
        outputs = model(torch.rand(2, 3, 4, 64, 64))

        self.assertEqual(model.encoder.backbone_name, "r2plus1d_18")
        self.assertEqual(outputs["features"].shape[0], 2)
        self.assertEqual(outputs["features"].ndim, 2)
        self.assertEqual(outputs["projections"].shape, (2, 16))

    def test_build_model_rejects_video_supcon_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "supcon.pt"
            torch.save(
                {
                    "checkpoint_type": "supcon_video_pretraining",
                    "encoder": {"encoder.weight": torch.tensor([1.0])},
                    "projection_head": {"projection_head.weight": torch.tensor([1.0])},
                },
                checkpoint_path,
            )

            config = {
                "model": {
                    "backbone": "resnet18",
                    "pretrained_backbone": False,
                    "pretrained_checkpoint": str(checkpoint_path),
                }
            }

            with self.assertRaisesRegex(ValueError, "not directly compatible"):
                build_model(config, num_keypoints=18)

    def test_run_supcon_training_writes_checkpoint_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            video_root = base / "videos"
            _write_test_video(video_root / "athlete01" / "session01" / "蛙_01.avi", frame_count=8)
            _write_test_video(video_root / "athlete02" / "session01" / "仰_01.avi", frame_count=8)
            index_path, summary = build_supcon_video_index(video_root, base / "supcon.csv")
            self.assertEqual(summary["valid"], 2)

            config_path = base / "supcon.toml"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    [experiment]
                    name = "supcon-smoke"
                    output_dir = "{(base / 'artifacts').as_posix()}"
                    seed = 7

                    [dataset]
                    video_index = "{index_path.as_posix()}"
                    input_width = 64
                    input_height = 64
                    clip_length = 4
                    frame_stride = 1
                    temporal_jitter = 0
                    crop_scale_min = 0.8
                    crop_scale_max = 1.0
                    color_jitter_strength = 0.1
                    grayscale_prob = 0.0
                    blur_prob = 0.0
                    blur_kernel_size = 3

                    [model]
                    backbone = "r2plus1d_18"
                    pretrained_backbone = false
                    pretrained_checkpoint = ""
                    projection_hidden_dim = 64
                    projection_dim = 16

                    [training]
                    epochs = 1
                    batch_size = 2
                    gradient_accumulation_steps = 1
                    learning_rate = 0.001
                    weight_decay = 0.0001
                    momentum = 0.9
                    optimizer = "sgd"
                    temperature = 0.07
                    warmup_epochs = 0
                    amp = false
                    clip_grad_norm = 0.0
                    num_workers = 0
                    device = "cpu"
                    use_cosine_schedule = false
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            checkpoint = run_supcon_training(config_path)
            metrics_path = base / "artifacts" / "train_metrics.json"

            self.assertTrue(checkpoint.exists())
            self.assertTrue(metrics_path.exists())
            state = torch.load(checkpoint, map_location="cpu")
            self.assertEqual(state["checkpoint_type"], "supcon_video_pretraining")
            self.assertEqual(state["encoder_backbone"], "r2plus1d_18")
            self.assertEqual(state["reuse_targets"], ["video"])
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload), 1)


def _write_test_video(path: Path, frame_count: int = 6, size: tuple[int, int] = (64, 64)) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise unittest.SkipTest("opencv-python is required for video smoke tests") from exc

    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 5.0, size)
    if not writer.isOpened():
        raise unittest.SkipTest("OpenCV video writer is unavailable in this environment")
    try:
        for index in range(frame_count):
            frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            frame[:, :, 0] = (index * 15) % 255
            frame[:, :, 1] = (50 + index * 10) % 255
            frame[:, :, 2] = (100 + index * 5) % 255
            writer.write(frame)
    finally:
        writer.release()


if __name__ == "__main__":
    unittest.main()
