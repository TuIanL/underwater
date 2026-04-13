from __future__ import annotations

import contextlib
import tempfile
import unittest
from pathlib import Path

from swim_pose.manifest import migrate_manifest_paths


class ManifestMigrationTests(unittest.TestCase):
    def test_manifest_migration_converts_legacy_external_relatives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo_root = base / "repo"
            repo_root.mkdir()
            (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
            manifests = repo_root / "data" / "manifests"
            manifests.mkdir(parents=True)
            legacy_root = base / "legacy"
            legacy_root.mkdir()
            original_cwd = legacy_root / "runs"
            original_cwd.mkdir()
            external_video = legacy_root / "videos" / "clip.mp4"
            external_video.parent.mkdir(parents=True)
            external_video.write_bytes(b"")
            internal_video = repo_root / "data" / "raw" / "videos" / "clip_inside.mp4"
            internal_video.parent.mkdir(parents=True)
            internal_video.write_bytes(b"")
            manifest_path = manifests / "clips.csv"
            manifest_path.write_text(
                "\n".join(
                    [
                        "clip_id,athlete_id,session_id,raw_above_path,raw_under_path,stitched_path,primary_view,sync_status,sync_offset_ms,fps_above,fps_under,frame_count_above,frame_count_under,duration_above_s,duration_under_s,notes",
                        "clip01,athlete01,session01,,,../videos/clip.mp4,stitched,pending_audit,,,,,,,",
                        "clip02,athlete01,session01,,,data/raw/videos/clip_inside.mp4,stitched,pending_audit,,,,,,,",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with contextlib.chdir(repo_root):
                destination, summary = migrate_manifest_paths(
                    path=manifest_path,
                    output_path=manifests / "clips.migrated.csv",
                    legacy_base=original_cwd,
                )

            self.assertEqual(summary["rows"], 2)
            self.assertEqual(summary["updated_fields"], 1)
            lines = destination.read_text(encoding="utf-8").splitlines()
            self.assertIn(str(external_video.resolve()), lines[1])
            self.assertIn("data/raw/videos/clip_inside.mp4", lines[2])


if __name__ == "__main__":
    unittest.main()
