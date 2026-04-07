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

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

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
# Ollama mock fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_ollama_client() -> MagicMock:
    """Create a mock Ollama client that returns sample analysis JSON.

    This fixture patches the ollama.Client.chat() method to return a
    predefined JSON response instead of calling the real Ollama API.

    Usage:
        def test_something(mock_ollama_client):
            # mock_ollama_client is automatically configured
            agent = CuratorAgent(ollama_client=mock_ollama_client)
    """

    def _make_client(response_data: dict[str, Any] | None = None) -> MagicMock:
        """Factory function to create a client with custom response data."""
        client = MagicMock()
        client.chat.return_value = {
            "message": {"content": json.dumps(response_data or SAMPLE_ANALYSIS)}
        }
        return client

    return _make_client


@pytest.fixture
def mock_ollama_client_json_chatter() -> MagicMock:
    """Create a mock Ollama client that returns JSON wrapped in markdown.

    This simulates a common LLM behavior: wrapping JSON in ```json blocks.
    The regex extractor should handle this correctly.
    """
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
    """Create a mock that fails twice then succeeds.

    This tests the retry loop: the first two calls return invalid JSON,
    the third call returns valid JSON.
    """
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

    The fake files are empty but have the correct extensions so that
    the file walker can find them.

    Returns:
        Path to the temporary folder containing media files.
    """
    folder = tmp_path / "test_photos"
    folder.mkdir()

    # Create fake image files (empty but with correct extensions)
    (folder / "IMG_001.jpg").write_bytes(b"fake jpg data")
    (folder / "IMG_002.jpg").write_bytes(b"fake jpg data longer" * 100)
    (folder / "IMG_003.png").write_bytes(b"fake png data")
    (folder / "DSC_001.arw").write_bytes(b"fake arw data")
    (folder / "video_001.mp4").write_bytes(b"fake mp4 data")

    # Create macOS metadata files (should be filtered out)
    (folder / "._IMG_001.jpg").write_bytes(b"macos metadata")

    return folder


@pytest.fixture
def temp_multiple_folders(tmp_path: Path) -> list[Path]:
    """Create multiple temporary folders with media files.

    This is useful for testing duplicate detection across folders.

    Returns:
        List of folder paths, each containing media files.
    """
    folders: list[Path] = []

    for i, name in enumerate(["vacation", "backup", "old_photos"]):
        folder = tmp_path / name
        folder.mkdir()
        folders.append(folder)

        # Each folder gets some unique files
        (folder / f"photo_{i}_001.jpg").write_bytes(b"unique data" * (i + 1))
        (folder / f"photo_{i}_002.jpg").write_bytes(b"unique data" * (i + 2))

        # All folders share one duplicate file (same content)
        (folder / "duplicate.jpg").write_bytes(b"same content everywhere")

    return folders


@pytest.fixture
def temp_folder_with_subdirs(tmp_path: Path) -> Path:
    """Create a folder with subdirectories containing media files.

    Tests that the LibrarianAgent correctly walks into subdirectories
    and groups files by their immediate parent folder.
    """
    root = tmp_path / "photo_library"
    root.mkdir()

    # Files in root
    (root / "root_photo.jpg").write_bytes(b"root photo")

    # Subfolder with its own files
    subdir = root / "2024_trip"
    subdir.mkdir()
    (subdir / "trip_001.jpg").write_bytes(b"trip photo")
    (subdir / "trip_002.jpg").write_bytes(b"trip photo 2")

    # Nested subfolder
    nested = subdir / "day1"
    nested.mkdir()
    (nested / "day1_001.jpg").write_bytes(b"day1 photo")

    # __MACOSX directory (should be filtered out)
    macosx = root / "__MACOSX"
    macosx.mkdir()
    (macosx / "ghost_file.jpg").write_bytes(b"should be ignored")

    return root


# ---------------------------------------------------------------------------
# Mocked external libraries
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_rawpy() -> MagicMock:
    """Mock the rawpy library for ARW file processing.

    This prevents tests from needing actual ARW files or the rawpy
    native library, which can be difficult to install in CI.
    """
    with patch("media_pruner.agent_librarian.rawpy") as mock:
        # Set up the context manager return value
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_context)
        mock_context.__exit__ = MagicMock(return_value=False)
        mock.imread.return_value = mock_context
        mock_context.postprocess.return_value = b"fake rgb data"
        yield mock


@pytest.fixture
def mock_cv2() -> MagicMock:
    """Mock the OpenCV library for video frame extraction."""
    with patch("media_pruner.agent_librarian.cv2") as mock:
        # Set up VideoCapture mock
        cap_mock = MagicMock()
        cap_mock.isOpened.return_value = True
        cap_mock.get.return_value = 100  # 100 total frames
        cap_mock.read.return_value = (True, b"fake frame data")
        mock.VideoCapture.return_value = cap_mock
        mock.COLOR_BGR2RGB = 4  # Constant value
        mock.CAP_PROP_FRAME_COUNT = 7
        mock.CAP_PROP_POS_FRAMES = 0
        yield mock


@pytest.fixture
def mock_subprocess() -> MagicMock:
    """Mock subprocess.run for AppleScript and Finder operations."""
    with patch("media_pruner.agent_decision.subprocess.run") as mock:
        mock.return_value = MagicMock(returncode=0, stdout="", stderr="")
        yield mock
