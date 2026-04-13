from __future__ import annotations

from pathlib import Path


class RepositoryContextError(RuntimeError):
    """Raised when repository-relative behavior is requested without repo context."""


def find_repo_root(start: str | Path | None = None) -> Path | None:
    current = Path(start) if start is not None else Path.cwd()
    current = current.expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def require_repo_root(start: str | Path | None = None) -> Path:
    root = find_repo_root(start)
    if root is None:
        raise RepositoryContextError(
            "Could not determine the repository root from the current working directory. "
            "Run this command from the repository root or a subdirectory inside it, or use "
            "absolute paths for external assets."
        )
    return root


def is_within(path: str | Path, root: str | Path) -> bool:
    candidate = Path(path).expanduser().resolve()
    boundary = Path(root).expanduser().resolve()
    try:
        candidate.relative_to(boundary)
    except ValueError:
        return False
    return True


def resolve_repo_managed_path(value: str | Path, repo_root: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    root = repo_root or require_repo_root()
    return (root / path).resolve()


def resolve_source_input_path(value: str | Path, repo_root: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    root = repo_root or find_repo_root()
    if root is not None:
        repo_candidate = (root / path).resolve()
        if repo_candidate.exists():
            return repo_candidate
    return cwd_candidate


def resolve_persisted_source_path(value: str | Path, repo_root: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    root = repo_root or require_repo_root()
    candidate = (root / path).resolve()
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Relative source asset path '{value}' was not found under repository root '{root}'. "
        "If this path was captured relative to another working directory and points outside the "
        "repository, migrate the manifest with `swim-pose manifest migrate-paths --legacy-base "
        "<dir> ...` or regenerate it."
    )


def serialize_workspace_path(value: str | Path, repo_root: Path | None = None) -> str:
    resolved = Path(value).expanduser().resolve()
    root = (repo_root.resolve() if repo_root is not None else find_repo_root(resolved))
    if root is not None and is_within(resolved, root):
        return resolved.relative_to(root).as_posix()
    return str(resolved)
