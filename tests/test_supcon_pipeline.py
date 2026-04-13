from __future__ import annotations

import json
import tempfile
import textwrap
import unittest
from pathlib import Path

from swim_pose.io import read_csv_rows
from swim_pose.manifest import (
    build_supcon_video_index,
    inspect_supcon_video,
    parse_supcon_video_stem,
)
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
                    backbone = "resnet18"
                    pretrained_backbone = false
                    pretrained_checkpoint = ""
                    projection_hidden_dim = 128
                    projection_dim = 32

                    [training]
                    epochs = 1
                    batch_size = 2
                    learning_rate = 0.001
                    weight_decay = 0.0001
                    momentum = 0.9
                    optimizer = "sgd"
                    temperature = 0.07
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
