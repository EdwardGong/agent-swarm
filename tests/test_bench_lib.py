"""Tests for shared bench-script utilities (`scripts/_bench_lib.py`).

Run with:  PYTHONPATH=. pytest tests/test_bench_lib.py -v

The bench helper module lives under scripts/ as a sibling-imported module
rather than a package, so we adjust sys.path before importing.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _bench_lib import display_path  # noqa: E402


class TestDisplayPath:
    """Regression coverage for the relative_to(REPO_ROOT) ValueError that
    crashed bench scripts when invoked with a relative --output-dir.

    The fix in `scripts/_bench_lib.py:display_path` resolves the input,
    returns it relative to ``root`` when possible, and falls back to the
    absolute path when it isn't. Each test below exercises one branch of
    that decision.
    """

    def test_relative_path_under_root(self, tmp_path, monkeypatch):
        """A relative path that resolves under root should display as a
        repo-relative Path. Pre-fix this raised ValueError because
        Path('reports/...').relative_to(Path('/abs/repo')) doesn't
        understand the relative input.
        """
        # Build an actual file under tmp_path so resolve() has a stable
        # anchor; chdir tmp_path so a relative input resolves under it.
        sub = tmp_path / "reports" / "benchmarks"
        sub.mkdir(parents=True)
        (sub / "out.json").write_text("{}")

        monkeypatch.chdir(tmp_path)
        result = display_path("reports/benchmarks/out.json", tmp_path)
        assert result == Path("reports/benchmarks/out.json")
        assert not result.is_absolute()

    def test_absolute_path_under_root(self, tmp_path):
        """An absolute path that lives under root should display relative."""
        sub = tmp_path / "a" / "b"
        sub.mkdir(parents=True)
        target = sub / "out.json"
        target.write_text("{}")

        result = display_path(target, tmp_path)
        assert result == Path("a/b/out.json")
        assert not result.is_absolute()

    def test_absolute_path_outside_root_falls_back(self, tmp_path):
        """An absolute path that does NOT live under root should fall back
        to the absolute path. This is the common case when the user passes
        --output-dir pointing somewhere outside the repo entirely.
        """
        # Use a sibling tmp dir so we know it's outside root.
        outside = tmp_path.parent / (tmp_path.name + "_outside")
        outside.mkdir()
        try:
            target = outside / "out.json"
            target.write_text("{}")

            result = display_path(target, tmp_path)
            assert result.is_absolute()
            assert result == target.resolve()
        finally:
            target.unlink(missing_ok=True)
            outside.rmdir()

    def test_relative_path_outside_root_falls_back(self, tmp_path, monkeypatch):
        """When a relative input resolves outside root (the original bug
        shape: cwd != repo root), display_path must not raise; it should
        fall back to the resolved absolute path.
        """
        # cwd is tmp_path; root is a sibling that the relative path won't
        # land under.
        outside_root = tmp_path.parent / (tmp_path.name + "_root")
        outside_root.mkdir()
        try:
            (tmp_path / "out.json").write_text("{}")
            monkeypatch.chdir(tmp_path)

            result = display_path("out.json", outside_root)
            assert result.is_absolute()
            assert result == (tmp_path / "out.json").resolve()
        finally:
            (tmp_path / "out.json").unlink(missing_ok=True)
            outside_root.rmdir()

    def test_accepts_str_input(self, tmp_path):
        """display_path should accept str as well as Path inputs."""
        target = tmp_path / "out.json"
        target.write_text("{}")
        result = display_path(str(target), tmp_path)
        assert result == Path("out.json")
