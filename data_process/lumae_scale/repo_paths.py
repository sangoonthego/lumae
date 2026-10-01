"""Centralized repository-relative path resolver and portability manager.

Ensures metadata artifacts store portable relative paths (e.g., local_data/raw/...)
rather than machine-specific absolute paths (e.g., D:\\... or /home/...).
"""

from __future__ import annotations

import os
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Optional, Union

from .models import ROOT as DEFAULT_REPO_ROOT


class PathResolver:
    """Resolves and normalizes canonical repository-relative paths."""

    def __init__(self, repo_root: Optional[Union[Path, str, PurePath]] = None) -> None:
        raw = repo_root if repo_root is not None else DEFAULT_REPO_ROOT
        self.raw_root_str = str(raw).replace("\\", "/")
        self.is_simulated_posix = self.raw_root_str.startswith("/") and not (":" in self.raw_root_str)
        if self.is_simulated_posix:
            self.repo_root = PurePosixPath(self.raw_root_str)
        else:
            self.repo_root = Path(raw).resolve()

    def to_relative(self, path: Union[Path, str, PurePath, None]) -> Optional[str]:
        """Convert an absolute or relative path to a canonical repo-relative POSIX string.

        Returns:
            A string with forward slashes (e.g. 'local_data/raw/adsqa/videos/abc.mp4'),
            or None if input is None.
        """
        if path is None:
            return None

        path_str = str(path).strip()
        if not path_str:
            return ""

        # Normalize slashes
        norm_str = path_str.replace("\\", "/")

        # Check if already relative
        is_abs = (
            ":" in norm_str
            or norm_str.startswith("//")
            or (norm_str.startswith("/") and not norm_str.startswith("local_data"))
        )

        if not is_abs:
            pure = PurePosixPath(norm_str)
            parts = [part for part in pure.parts if part not in (".", "")]
            return PurePosixPath(*parts).as_posix()

        # Path is absolute: attempt to resolve relative to repo_root
        if "local_data/" in norm_str:
            idx = norm_str.index("local_data/")
            return norm_str[idx:]

        if not self.is_simulated_posix:
            try:
                resolved_p = Path(path_str).resolve()
                rel = resolved_p.relative_to(self.repo_root)
                return rel.as_posix()
            except ValueError:
                pass

        # If not relative to repo_root (e.g. temporary test directories), return normalized path
        return norm_str

    def resolve(self, path: Union[Path, str, PurePath, None]) -> Optional[Union[Path, PurePosixPath]]:
        """Resolve a relative or absolute path against repo_root.

        Supports POSIX and Windows path strings seamlessly.
        """
        if path is None:
            return None

        path_str = str(path).strip()
        if not path_str:
            return self.repo_root

        # Normalize relative path (replace any backslashes)
        clean_rel = path_str.replace("\\", "/").lstrip("/")

        # Check if path is already absolute with respect to current host OS
        if not self.is_simulated_posix:
            p = Path(path_str)
            if p.is_absolute():
                return p.resolve()
            return (self.repo_root / clean_rel).resolve()
        else:
            if clean_rel.startswith(self.raw_root_str.lstrip("/")):
                return PurePosixPath("/" + clean_rel)
            return PurePosixPath(self.raw_root_str) / clean_rel

    def is_repo_owned(self, path: Union[Path, str, PurePath]) -> bool:
        """Return True if path is contained within the repository root."""
        try:
            rel = self.to_relative(path)
            return rel is not None and not rel.startswith("../")
        except ValueError:
            return False

    def is_canonical_relative(self, path_str: str) -> bool:
        """Check if a path string is already canonical (relative, forward slashes, no drive letters)."""
        if not path_str:
            return False
        if ":" in path_str or path_str.startswith("\\\\") or path_str.startswith("/"):
            return False
        if "\\" in path_str:
            return False
        return True


# Default instance
_DEFAULT_RESOLVER = PathResolver()


def to_repo_relative(path: Union[Path, str, PurePath, None], root: Optional[Union[Path, str, PurePath]] = None) -> Optional[str]:
    """Convert path to canonical repo-relative string."""
    resolver = PathResolver(root) if root else _DEFAULT_RESOLVER
    return resolver.to_relative(path)


def resolve_path(path: Union[Path, str, PurePath, None], root: Optional[Union[Path, str, PurePath]] = None) -> Optional[Union[Path, PurePosixPath]]:
    """Resolve path against repository root."""
    resolver = PathResolver(root) if root else _DEFAULT_RESOLVER
    return resolver.resolve(path)


def is_canonical_relative(path_str: str) -> bool:
    """Return True if path_str is canonical repo-relative."""
    return _DEFAULT_RESOLVER.is_canonical_relative(path_str)
