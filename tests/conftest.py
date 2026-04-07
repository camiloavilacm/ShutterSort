"""Shared pytest fixtures for the ShutterSort test suite.

Fixtures are reusable test setup code. Instead of duplicating setup logic
in every test function, we define it once here and pytest injects it
automatically based on the parameter name.

This is a powerful pattern because:
    - Fixtures are lazy (only created when a test needs them)
    - They support setup/teardown via yield
    - They can be scoped (function, class, module, session)
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Sample analysis data (what the LLM would return)
# ---------------------------------------------------------------------------
SAMPLE_ANALYSIS = {
    "scene_type": "landscape",
    "score": 8,
    "summary": "Beautiful mountain vista at golden hour",
    "people_count": 0,
    "people_description": "",
    "emotions_detected": "peaceful, serene",
}

SAMPLE_ANALYSIS_PORTRAIT = {
    "scene_type": "portrait",
    "score": 7,
    "summary": "Family group photo at a birthday party",
    "people_count": 4,
    "people_description": "Two adults and two children, casual clothing, smiling",
    "emotions_detected": "happy, celebratory",
}

SAMPLE_ANALYSIS_JUNK = {
    "scene_type": "junk",
    "score": 2,
    "summary": "Blurry screenshot of a receipt",
    "people_count": 0,
    "people_description": "",
    "emotions_detected": "",
}


# ---------------------------------------------------------------------------
# Helpers: create real minimal image files that PIL can open
# ---------------------------------------------------------------------------
def make_fake_jpeg(path: Path, size_multiplier: int = 1) -> None:
    """Create a minimal valid JPEG file that PIL can open."""
    from PIL import Image

    img = Image.new("RGB", (100, 100), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    path.write_bytes(buf.getvalue() * size_multiplier)


def make_fake_png(path: Path) -> None:
    """Create a minimal valid PNG file."""
    from PIL import Image

    img = Image.new("RGBA", (100, 100), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    path.write_bytes(buf.getvalue())


def make_fake_arw(path: Path) -> None:
    """Create a fake ARW file (rawpy will be mocked in tests)."""
    path.write_bytes(b"fake arw header" + b"\x00" * 100)


def make_fake_mp4(path: Path) -> None:
    """Create a fake MP4 file (cv2 will be mocked in tests)."""
    path.write_bytes(b"\x00\x00\x00\x1cftypmp42" + b"\x00" * 100)


# ---------------------------------------------------------------------------
# Ollama mock fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_ollama_client() -> Any:
    """Create a mock Ollama client that returns sample analysis JSON.

    This fixture returns a factory function so each test can customize
    the response data.

    Usage:
        def test_something(mock_ollama_client):
            client = mock_ollama_client()  # Uses default SAMPLE_ANALYSIS
            client = mock_ollama_client({"scene_type": "portrait"})  # Custom
    """

    def _make_client(response_data: dict[str, Any] | None = None) -> MagicMock:
        client = MagicMock()
        client.chat.return_value = {
            "message": {"content": json.dumps(response_data or SAMPLE_ANALYSIS)}
        }
        return client

    return _make_client


@pytest.fixture
def mock_ollama_client_json_chatter() -> MagicMock:
    """Create a mock Ollama client that returns JSON wrapped in markdown."""
    client = MagicMock()
    client.chat.return_value = {
        "message": {
            "content": (
                "Here's my analysis:\n\n"
                "```json\n"
                f"{json.dumps(SAMPLE_ANALYSIS)}\n"
                "```\n\n"
                "Hope this helps!"
            )
        }
    }
    return client


@pytest.fixture
def mock_ollama_client_fail_then_succeed() -> MagicMock:
    """Create a mock that fails twice then succeeds (tests retry loop)."""
    client = MagicMock()
    client.chat.side_effect = [
        {"message": {"content": "Invalid response 1"}},
        {"message": {"content": "Invalid response 2"}},
        {"message": {"content": json.dumps(SAMPLE_ANALYSIS)}},
    ]
    return client


@pytest.fixture
def mock_ollama_client_always_fails() -> MagicMock:
    """Create a mock that always returns invalid JSON."""
    client = MagicMock()
    client.chat.return_value = {"message": {"content": "This is not JSON at all!"}}
    return client


# ---------------------------------------------------------------------------
# Temporary directory with fake media files
# ---------------------------------------------------------------------------
@pytest.fixture
def temp_media_folder(tmp_path: Path) -> Path:
    """Create a temporary folder with fake media files for testing.

    Uses pytest's tmp_path fixture which creates a unique temporary
    directory for each test function and cleans it up automatically.
    """
    folder = tmp_path / "test_photos"
    folder.mkdir()

    make_fake_jpeg(folder / "IMG_001.jpg")
    make_fake_jpeg(folder / "IMG_002.jpg", size_multiplier=100)
    make_fake_png(folder / "IMG_003.png")
    make_fake_arw(folder / "DSC_001.arw")
    make_fake_mp4(folder / "video_001.mp4")
    (folder / "._IMG_001.jpg").write_bytes(b"macos metadata")

    return folder


@pytest.fixture
def temp_multiple_folders(tmp_path: Path) -> list[Path]:
    """Create multiple temporary folders with media files for duplicate testing."""
    folders: list[Path] = []

    for i, name in enumerate(["vacation", "backup", "old_photos"]):
        folder = tmp_path / name
        folder.mkdir()
        folders.append(folder)

        make_fake_jpeg(folder / f"photo_{i}_001.jpg", size_multiplier=i + 1)
        make_fake_jpeg(folder / f"photo_{i}_002.jpg", size_multiplier=i + 2)
        make_fake_jpeg(folder / "duplicate.jpg")

    return folders


@pytest.fixture
def temp_folder_with_subdirs(tmp_path: Path) -> Path:
    """Create a folder with subdirectories containing media files."""
    root = tmp_path / "photo_library"
    root.mkdir()

    make_fake_jpeg(root / "root_photo.jpg")

    subdir = root / "2024_trip"
    subdir.mkdir()
    make_fake_jpeg(subdir / "trip_001.jpg")
    make_fake_jpeg(subdir / "trip_002.jpg")

    nested = subdir / "day1"
    nested.mkdir()
    make_fake_jpeg(nested / "day1_001.jpg")

    macosx = root / "__MACOSX"
    macosx.mkdir()
    (macosx / "ghost_file.jpg").write_bytes(b"should be ignored")

    return root
