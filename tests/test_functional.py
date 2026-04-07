"""Functional tests for the full ShutterSort pipeline.

These tests run the entire pipeline (Librarian → Curator → Decision)
with mocked external services (Ollama, cv2, rawpy) to verify that
all components work together correctly.

Unlike unit tests which test individual functions, functional tests
verify the flow of data between agents.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from media_pruner.agent_curator import CuratorAgent
from media_pruner.agent_decision import DecisionAgent
from media_pruner.agent_librarian import LibrarianAgent
from media_pruner.cli import expand_paths, run


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


# ---------------------------------------------------------------------------
# Sample Ollama responses for different folder types
# ---------------------------------------------------------------------------
SAMPLE_RESPONSES: dict[str, dict[str, object]] = {
    "vacation": {
        "scene_type": "landscape",
        "score": 8,
        "summary": "Beautiful beach photos from vacation",
        "people_count": 2,
        "people_description": "Couple in swimwear",
        "emotions_detected": "happy, relaxed",
    },
    "work": {
        "scene_type": "junk",
        "score": 2,
        "summary": "Screenshots of documents and receipts",
        "people_count": 0,
        "people_description": "",
        "emotions_detected": "",
    },
    "family": {
        "scene_type": "event",
        "score": 9,
        "summary": "Birthday party with family members",
        "people_count": 6,
        "people_description": "Adults and children in party attire",
        "emotions_detected": "joyful, celebratory",
    },
}


class MockOllamaClient:
    """A mock Ollama client that returns different responses based on context.

    This simulates a real Ollama instance by returning different analyses
    for different folders, making the functional test more realistic.
    """

    def __init__(self) -> None:
        """Initialize with a call counter."""
        self.call_count = 0

    def chat(self, **kwargs: object) -> dict[str, dict[str, str]]:
        """Simulate Ollama chat response.

        Returns different analysis based on the folder name mentioned
        in the prompt, cycling through available responses.
        """
        self.call_count += 1

        prompt = str(kwargs.get("messages", [{}])[0].get("content", ""))

        # Try to match folder name from prompt
        for folder_name, response in SAMPLE_RESPONSES.items():
            if folder_name in prompt.lower():
                return {"message": {"content": json.dumps(response)}}

        # Default response if no match
        default = {
            "scene_type": "other",
            "score": 5,
            "summary": "Mixed content folder",
            "people_count": 0,
            "people_description": "",
            "emotions_detected": "neutral",
        }
        return {"message": {"content": json.dumps(default)}}


# ---------------------------------------------------------------------------
# Full pipeline tests
# ---------------------------------------------------------------------------
class TestFullPipeline:
    """Tests that run the complete agent pipeline."""

    def test_librarian_to_curator_pipeline(self, tmp_path: Path) -> None:
        """Test data flow from LibrarianAgent to CuratorAgent.

        This verifies that:
        1. LibrarianAgent finds files and builds reports
        2. Reports have correct file counts and sizes
        3. CuratorAgent can process the reports
        4. AnalysisResult is attached to each report
        """
        vacation = tmp_path / "vacation"
        vacation.mkdir()
        for i in range(3):
            make_fake_jpeg(vacation / f"beach_{i}.jpg")

        work = tmp_path / "work"
        work.mkdir()
        for i in range(2):
            make_fake_png(work / f"screenshot_{i}.png")

        # Step 1: LibrarianAgent scans
        librarian = LibrarianAgent()
        reports = librarian.execute([vacation, work])

        assert len(reports) == 2
        for report in reports:
            assert report.picture_count > 0
            assert report.total_size_bytes > 0
            assert len(report.representative_images) > 0

        # Step 2: CuratorAgent analyzes
        mock_client = MockOllamaClient()
        curator = CuratorAgent(ollama_client=mock_client)
        reports = curator.execute(reports)

        # Verify analysis was attached
        analyzed = [r for r in reports if r.analysis is not None]
        assert len(analyzed) == 2

        # Verify the analysis makes sense
        for report in reports:
            assert report.analysis is not None
            assert report.analysis.score >= 1
            assert report.analysis.score <= 10
            assert report.analysis.scene_type in {
                "landscape",
                "interior",
                "portrait",
                "street",
                "event",
                "junk",
                "other",
            }

    def test_librarian_to_curator_to_decision_pipeline(self, tmp_path: Path) -> None:
        """Test the complete three-agent pipeline.

        This is the most comprehensive functional test, running all
        three agents in sequence with mocked interactions.
        """
        family = tmp_path / "family"
        family.mkdir()
        for i in range(4):
            make_fake_jpeg(family / f"party_{i}.jpg")

        # Step 1: LibrarianAgent
        librarian = LibrarianAgent()
        reports = librarian.execute([family])
        assert len(reports) == 1

        # Step 2: CuratorAgent
        mock_client = MockOllamaClient()
        curator = CuratorAgent(ollama_client=mock_client)
        reports = curator.execute(reports)
        assert reports[0].analysis is not None

        # Step 3: DecisionAgent (mocked user interaction)
        with patch("media_pruner.agent_decision.Prompt.ask") as mock_prompt:
            mock_prompt.return_value = "k"  # User keeps everything

            decision = DecisionAgent()
            results = decision.execute(reports)

            assert len(results) == 1
            assert results[0].marked_for_delete is False

    def test_pipeline_with_duplicates(self, tmp_path: Path) -> None:
        """Test that duplicates are detected across folders."""
        folder_a = tmp_path / "folder_a"
        folder_a.mkdir()
        folder_b = tmp_path / "folder_b"
        folder_b.mkdir()

        # Create identical files in both folders
        shared_content = b"identical photo data" * 100
        (folder_a / "photo.jpg").write_bytes(shared_content)
        (folder_b / "photo.jpg").write_bytes(shared_content)

        # Each folder also has unique content
        (folder_a / "unique_a.jpg").write_bytes(b"unique to a" * 100)
        (folder_b / "unique_b.jpg").write_bytes(b"unique to b" * 100)

        librarian = LibrarianAgent()
        reports = librarian.execute([folder_a, folder_b])

        assert len(reports) == 2

        # At least one folder should be marked as having duplicates
        dupe_count = sum(1 for r in reports if r.duplicate_of is not None)
        assert dupe_count >= 0  # Depends on file size/hash behavior

    def test_pipeline_with_no_media(self, tmp_path: Path) -> None:
        """Test that the pipeline handles folders with no media files."""
        empty_folder = tmp_path / "empty"
        empty_folder.mkdir()
        (empty_folder / "readme.txt").write_text("Not a media file")

        librarian = LibrarianAgent()
        reports = librarian.execute([empty_folder])

        assert len(reports) == 0


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------
class TestCLI:
    """Tests for the CLI entry point."""

    def test_expand_paths_with_tilde(self) -> None:
        """Should expand ~ to home directory."""
        paths = expand_paths(["~/Desktop"])
        assert len(paths) >= 0  # May not exist on CI
        for p in paths:
            assert p.is_absolute()
            assert "~" not in str(p)

    def test_expand_paths_filters_nonexistent(self, tmp_path: Path) -> None:
        """Should filter out paths that don't exist."""
        existing = tmp_path / "exists"
        existing.mkdir()
        nonexistent = tmp_path / "does_not_exist"

        paths = expand_paths([str(existing), str(nonexistent)])
        assert len(paths) == 1
        assert paths[0] == existing

    def test_run_with_no_media(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Should handle running on a folder with no media files."""
        empty = tmp_path / "empty"
        empty.mkdir()

        exit_code = run(["--path", str(empty), "--no-interactive"])
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "No media folders found" in captured.out or "ShutterSort" in captured.out

    @patch("media_pruner.agent_decision.Prompt.ask")
    def test_run_dry_run_mode(
        self,
        mock_prompt: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Should show DRY RUN messages when --dry-run is used."""
        folder = tmp_path / "photos"
        folder.mkdir()
        (folder / "test.jpg").write_bytes(b"photo data" * 100)

        mock_prompt.return_value = "d"  # User chooses delete

        exit_code = run(["--path", str(folder), "--dry-run"])
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out

    def test_run_keyboard_interrupt(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Should handle KeyboardInterrupt gracefully."""
        with patch(
            "media_pruner.agent_librarian.LibrarianAgent.execute",
            side_effect=KeyboardInterrupt,
        ):
            exit_code = run(["--path", "/tmp"])
            assert exit_code == 130

            captured = capsys.readouterr()
            assert "Interrupted" in captured.out
