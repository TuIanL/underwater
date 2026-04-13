from __future__ import annotations

import contextlib
import tempfile
import unittest
from pathlib import Path

from swim_pose.pathing import (
    find_repo_root,
    resolve_repo_managed_path,
    resolve_source_input_path,
    serialize_workspace_path,
)


class PathingTests(unittest.TestCase):
    def test_repo_managed_paths_resolve_from_repo_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
            nested = repo_root / "src" / "pkg"
            nested.mkdir(parents=True)
            target = repo_root / "configs" / "demo.toml"
            target.parent.mkdir(parents=True)
            target.write_text("", encoding="utf-8")

            with contextlib.chdir(nested):
                resolved = resolve_repo_managed_path("configs/demo.toml")

            self.assertEqual(resolved, target.resolve())

    def test_source_inputs_prefer_existing_cwd_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
            caller_root = Path(tmp) / "caller"
            caller_root.mkdir()
            external = caller_root / "videos"
            external.mkdir()

            with contextlib.chdir(repo_root):
                resolved = resolve_source_input_path("../caller/videos")

            self.assertEqual(resolved, external.resolve())

    def test_workspace_paths_serialize_relative_inside_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
            internal = repo_root / "data" / "frames" / "clip.jpg"
            internal.parent.mkdir(parents=True)
            internal.write_text("", encoding="utf-8")

            self.assertEqual(
                serialize_workspace_path(internal, repo_root),
                "data/frames/clip.jpg",
            )
            self.assertEqual(find_repo_root(repo_root / "data"), repo_root.resolve())


if __name__ == "__main__":
    unittest.main()
