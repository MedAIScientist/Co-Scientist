"""Path-traversal hardening tests for storage.artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

from co_scientist.config import Config
from co_scientist.storage import artifacts


def _cfg(tmp_path: Path) -> Config:
    c = Config()
    c.storage.data_dir = str(tmp_path)
    return c


@pytest.mark.asyncio
async def test_write_json_rejects_dotdot_in_id(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError):
        await artifacts.write_json(cfg, "sess1", "transcripts", "../../etc/passwd", {})


@pytest.mark.asyncio
async def test_write_json_rejects_dotdot_in_kind(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError):
        await artifacts.write_json(cfg, "sess1", "../escape", "abc", {})


@pytest.mark.asyncio
async def test_write_json_rejects_absolute_kind(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError):
        await artifacts.write_json(cfg, "sess1", "/etc", "abc", {})


@pytest.mark.asyncio
async def test_write_json_rejects_session_dotdot(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError):
        await artifacts.write_json(cfg, "..", "transcripts", "abc", {})


@pytest.mark.asyncio
async def test_write_json_accepts_slash_in_kind(tmp_path: Path) -> None:
    """Legitimate callers use kinds like 'transcripts/generation'."""
    cfg = _cfg(tmp_path)
    rel = await artifacts.write_json(cfg, "sess1", "transcripts/generation", "abc", {"k": 1})
    assert "transcripts/generation" in rel
    assert (tmp_path / rel).is_file()


@pytest.mark.asyncio
async def test_read_json_rejects_path_escape(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    # Create a target outside data_dir
    secret = tmp_path.parent / "secret.json"
    secret.write_text('{"x": 1}')
    with pytest.raises(ValueError):
        await artifacts.read_json(cfg, "../secret.json")
