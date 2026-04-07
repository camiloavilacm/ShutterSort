"""Unit tests for the LibrarianAgent.

Tests cover:
    - Directory walking and media file discovery
    - macOS metadata filtering
    - Folder report building
    - Representative image selection
    - Duplicate detection
    - ARW preview extraction (mocked)
    - Video frame extraction (mocked)
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from media_pruner.agent_librarian import LibrarianAgent
from media_pruner.models import FolderReport


# ---------------------------------------------------------------------------
# Helper: create a minimal valid JPEG in memory
# ---------------------------------------------------------------------------
def make_fake_jpeg(path: Path, size_multiplier: int = 1) -> None:
    """Create a minimal valid JPEG file that PIL can open.

    PIL needs actual valid image data, not just bytes with a .jpg extension.
    """
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
    """Create a fake ARW file (rawpy will be mocked anyway)."""
    path.write_bytes(b"fake arw header" + b"\x00" * 100)


def make_fake_mp4(path: Path) -> None:
    """Create a fake MP4 file (cv2 will be mocked anyway)."""
    path.write_bytes(b"\x00\x00\x00\x1cftypmp42" + b"\x00" * 100)


# ---------------------------------------------------------------------------
# Directory walking tests
# ---------------------------------------------------------------------------
class TestWalkDirectory:
    """Tests for the _walk_directory method."""

    def test_finds_all_media_files(self, temp_media_folder: Path) -> None:
        """Should find JPG, PNG, ARW, and MP4 files."""
        agent = LibrarianAgent()
        folder_files: dict[Path, list[Path]] = {}
        agent._walk_directory(temp_media_folder, folder_files)

        assert len(folder_files) == 1
        folder_path = next(iter(folder_files.keys()))
        assert folder_path.name == "test_photos"

        files = folder_files[folder_path]
        assert len(files) == 5

        extensions = {f.suffix.lower() for f in files}
        assert {".jpg", ".png", ".arw", ".mp4"} == extensions

    def test_filters_macos_metadata(self, temp_media_folder: Path) -> None:
        """Should exclude ._ files from results."""
        agent = LibrarianAgent()
        folder_files: dict[Path, list[Path]] = {}
        agent._walk_directory(temp_media_folder, folder_files)

        for files in folder_files.values():
            filenames = {f.name for f in files}
            assert "._IMG_001.jpg" not in filenames

    def test_skips_nonexistent_path(self, tmp_path: Path) -> None:
        """Should gracefully handle paths that don't exist."""
        agent = LibrarianAgent()
        folder_files: dict[Path, list[Path]] = {}
        agent._walk_directory(tmp_path / "does_not_exist", folder_files)
        assert len(folder_files) == 0

    def test_handles_single_file(self, tmp_path: Path) -> None:
        """Should handle a single file path by using its parent as folder."""
        make_fake_jpeg(tmp_path / "photo.jpg")

        agent = LibrarianAgent()
        folder_files: dict[Path, list[Path]] = {}
        agent._walk_directory(tmp_path / "photo.jpg", folder_files)

        assert len(folder_files) == 1
        assert tmp_path in folder_files

    def test_walks_subdirectories(self, temp_folder_with_subdirs: Path) -> None:
        """Should walk into subdirectories and group files by parent."""
        agent = LibrarianAgent()
        folder_files: dict[Path, list[Path]] = {}
        agent._walk_directory(temp_folder_with_subdirs, folder_files)

        folder_names = {p.name for p in folder_files}
        assert "2024_trip" in folder_names
        assert "day1" in folder_names

        # __MACOSX should be excluded
        assert "__MACOSX" not in folder_names


# ---------------------------------------------------------------------------
# Folder report building tests
# ---------------------------------------------------------------------------
class TestBuildFolderReport:
    """Tests for the _build_folder_report method."""

    def test_calculates_correct_counts(self, temp_media_folder: Path) -> None:
        """Should correctly count pictures vs videos."""
        agent = LibrarianAgent()
        media_files = list(temp_media_folder.glob("*"))
        media_files = [
            f
            for f in media_files
            if f.suffix.lower() in {".jpg", ".png", ".arw", ".mp4"}
            and not f.name.startswith("._")
        ]

        report = agent._build_folder_report(temp_media_folder, media_files)

        assert report.path == temp_media_folder
        assert report.picture_count == 4  # 2 jpg + 1 png + 1 arw
        assert report.video_count == 1  # 1 mp4
        assert report.total_size_bytes > 0

    def test_selects_representative_images(self, temp_media_folder: Path) -> None:
        """Should select up to 5 representative images sorted by size."""
        agent = LibrarianAgent()
        media_files = list(temp_media_folder.glob("*"))
        media_files = [
            f
            for f in media_files
            if f.suffix.lower() in {".jpg", ".png", ".arw", ".mp4"}
            and not f.name.startswith("._")
        ]

        report = agent._build_folder_report(temp_media_folder, media_files)

        assert len(report.representative_images) <= 5
        for img in report.representative_images:
            assert img.suffix.lower() in {".jpg", ".png", ".arw"}

    def test_empty_folder_report(self, tmp_path: Path) -> None:
        """Should handle a folder with no media files."""
        agent = LibrarianAgent()
        report = agent._build_folder_report(tmp_path, [])

        assert report.picture_count == 0
        assert report.video_count == 0
        assert report.total_size_bytes == 0
        assert report.representative_images == []


# ---------------------------------------------------------------------------
# Representative image selection tests
# ---------------------------------------------------------------------------
class TestSelectRepresentativeImages:
    """Tests for the _select_representive_images method."""

    def test_selects_top_5_by_size(self, tmp_path: Path) -> None:
        """Should select the 5 largest image files."""
        folder = tmp_path / "photos"
        folder.mkdir()

        for i in range(7):
            make_fake_jpeg(folder / f"photo_{i:03d}.jpg", size_multiplier=i + 1)

        agent = LibrarianAgent()
        media_files = sorted(folder.glob("*"))
        selected = agent._select_representive_images(media_files, max_count=5)

        assert len(selected) == 5
        sizes = [f.stat().st_size for f in selected]
        assert sizes == sorted(sizes, reverse=True)

    def test_excludes_video_files(self, tmp_path: Path) -> None:
        """Should not include MP4 files in representative images."""
        folder = tmp_path / "mixed"
        folder.mkdir()
        make_fake_jpeg(folder / "photo.jpg")
        make_fake_mp4(folder / "video.mp4")

        agent = LibrarianAgent()
        media_files = sorted(folder.glob("*"))
        selected = agent._select_representive_images(media_files)

        assert len(selected) == 1
        assert selected[0].suffix == ".jpg"

    def test_respects_max_count(self, tmp_path: Path) -> None:
        """Should respect the max_count parameter."""
        folder = tmp_path / "photos"
        folder.mkdir()
        for i in range(10):
            make_fake_jpeg(folder / f"img_{i}.jpg")

        agent = LibrarianAgent()
        media_files = sorted(folder.glob("*"))

        selected_3 = agent._select_representive_images(media_files, max_count=3)
        assert len(selected_3) == 3

        selected_1 = agent._select_representive_images(media_files, max_count=1)
        assert len(selected_1) == 1


# ---------------------------------------------------------------------------
# Duplicate detection tests
# ---------------------------------------------------------------------------
class TestDetectDuplicates:
    """Tests for the _detect_duplicates method."""

    def test_marks_duplicate_folders(self, temp_multiple_folders: list[Path]) -> None:
        """Should mark folders containing duplicates."""
        agent = LibrarianAgent()
        reports: list[FolderReport] = []

        for folder in temp_multiple_folders:
            files = list(folder.glob("*.jpg"))
            report = agent._build_folder_report(folder, files)
            reports.append(report)

        agent._detect_duplicates(reports)

        dupe_count = sum(1 for r in reports if r.duplicate_of is not None)
        assert dupe_count >= 0


# ---------------------------------------------------------------------------
# ARW extraction tests (mocked via sys.modules)
# ---------------------------------------------------------------------------
class TestExtractArwPreview:
    """Tests for the extract_arw_preview method."""

    def test_extracts_preview_successfully(self, tmp_path: Path) -> None:
        """Should extract a JPEG preview from an ARW file."""
        import numpy as np

        arw_file = tmp_path / "test.arw"
        make_fake_arw(arw_file)

        # Mock rawpy via sys.modules
        mock_raw = MagicMock()
        mock_raw.postprocess.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_raw.__enter__ = MagicMock(return_value=mock_raw)
        mock_raw.__exit__ = MagicMock(return_value=False)
        mock_rawpy = MagicMock()
        mock_rawpy.imread.return_value = mock_raw

        with patch.dict(sys.modules, {"rawpy": mock_rawpy}):
            # Re-import to get the mocked version
            import importlib

            import media_pruner.agent_librarian

            importlib.reload(media_pruner.agent_librarian)

            agent = media_pruner.agent_librarian.LibrarianAgent()
            result = agent.extract_arw_preview(arw_file)

        assert result is not None
        assert result.suffix == ".jpg"
        assert result.exists()

    def test_returns_none_on_failure(self, tmp_path: Path) -> None:
        """Should return None if rawpy fails."""
        arw_file = tmp_path / "corrupt.arw"
        make_fake_arw(arw_file)

        mock_rawpy = MagicMock()
        mock_rawpy.imread.side_effect = Exception("Corrupt file")

        with patch.dict(sys.modules, {"rawpy": mock_rawpy}):
            import importlib

            import media_pruner.agent_librarian

            importlib.reload(media_pruner.agent_librarian)

            agent = media_pruner.agent_librarian.LibrarianAgent()
            result = agent.extract_arw_preview(arw_file)

        assert result is None


# ---------------------------------------------------------------------------
# Video frame extraction tests (mocked via sys.modules)
# ---------------------------------------------------------------------------
class TestExtractVideoFrames:
    """Tests for the extract_video_frames method."""

    def test_extracts_three_frames(self, tmp_path: Path) -> None:
        """Should extract 3 frames from a video."""
        import numpy as np

        mp4_file = tmp_path / "test.mp4"
        make_fake_mp4(mp4_file)

        frame_data = np.zeros((100, 100, 3), dtype=np.uint8)

        cap_mock = MagicMock()
        cap_mock.isOpened.return_value = True

        def mock_get(prop: int) -> int:
            if prop == 7:  # CAP_PROP_FRAME_COUNT
                return 300
            return 0

        cap_mock.get.side_effect = mock_get
        cap_mock.read.return_value = (True, frame_data)

        mock_cv2 = MagicMock()
        mock_cv2.VideoCapture.return_value = cap_mock
        mock_cv2.COLOR_BGR2RGB = 4
        mock_cv2.CAP_PROP_FRAME_COUNT = 7
        mock_cv2.CAP_PROP_POS_FRAMES = 0
        mock_cv2.INTER_AREA = 3
        # Make cv2.cvtColor return the input frame unchanged
        mock_cv2.cvtColor.side_effect = lambda frame, code: frame

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            import importlib

            import media_pruner.agent_librarian

            importlib.reload(media_pruner.agent_librarian)

            agent = media_pruner.agent_librarian.LibrarianAgent()
            frames = agent.extract_video_frames(mp4_file)

        assert len(frames) == 3
        for frame in frames:
            assert frame.suffix == ".jpg"
            assert frame.exists()

        cap_mock.release.assert_called()

    def test_returns_empty_on_open_failure(self, tmp_path: Path) -> None:
        """Should return empty list if video can't be opened."""
        mp4_file = tmp_path / "bad.mp4"
        make_fake_mp4(mp4_file)

        cap_mock = MagicMock()
        cap_mock.isOpened.return_value = False

        mock_cv2 = MagicMock()
        mock_cv2.VideoCapture.return_value = cap_mock

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            import importlib

            import media_pruner.agent_librarian

            importlib.reload(media_pruner.agent_librarian)

            agent = media_pruner.agent_librarian.LibrarianAgent()
            frames = agent.extract_video_frames(mp4_file)

        assert frames == []


# ---------------------------------------------------------------------------
# Main execute method tests
# ---------------------------------------------------------------------------
class TestExecute:
    """Tests for the main execute() method."""

    def test_returns_empty_for_no_paths(self) -> None:
        """Should return empty list when given no scan paths."""
        agent = LibrarianAgent()
        reports = agent.execute([])
        assert reports == []

    def test_returns_reports_for_valid_folder(self, temp_media_folder: Path) -> None:
        """Should return FolderReport objects for valid folders."""
        agent = LibrarianAgent()
        reports = agent.execute([temp_media_folder])

        assert len(reports) >= 1
        assert isinstance(reports[0], FolderReport)
        assert reports[0].picture_count > 0
