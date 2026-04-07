"""Unit tests for the DecisionAgent.

Tests cover:
    - Summary table display
    - Interactive review loop (keep, delete, skip, open)
    - Verification loop for high-score deletions
    - AppleScript trash operations (mocked)
    - Dry run mode
    - Actions summary
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from media_pruner.agent_decision import DecisionAgent
from media_pruner.models import AnalysisResult, FolderReport


# ---------------------------------------------------------------------------
# Helper to create a test FolderReport with analysis
# ---------------------------------------------------------------------------
def make_report(
    path: Path,
    scene_type: str = "landscape",
    score: int = 7,
    people_count: int = 0,
    media_count: int = 5,
    size_bytes: int = 1_000_000,
) -> FolderReport:
    """Create a FolderReport with an AnalysisResult for testing."""
    media_files = [path / f"img_{i}.jpg" for i in range(media_count)]
    return FolderReport(
        path=path,
        media_files=media_files,
        total_size_bytes=size_bytes,
        picture_count=media_count,
        video_count=0,
        analysis=AnalysisResult(
            scene_type=scene_type,  # type: ignore[arg-type]
            score=score,
            summary=f"A {scene_type} folder with {media_count} files",
            people_count=people_count,
            emotions_detected="neutral",
        ),
    )


# ---------------------------------------------------------------------------
# Summary table tests
# ---------------------------------------------------------------------------
class TestDisplaySummaryTable:
    """Tests for the _display_summary_table method."""

    def test_displays_empty_reports(self) -> None:
        """Should handle an empty reports list gracefully."""
        agent = DecisionAgent()
        agent._display_summary_table([])
        # Should not raise — Rich handles empty tables

    def test_displays_report_with_analysis(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        """Should display a report with full analysis."""
        folder = tmp_path / "vacation"
        folder.mkdir()
        report = make_report(folder, scene_type="landscape", score=8)

        agent = DecisionAgent()
        agent._display_summary_table([report])

        captured = capsys.readouterr()
        # The output should contain key information
        assert "vacation" in captured.out
        assert "lan" in captured.out  # "landscape" may be truncated


# ---------------------------------------------------------------------------
# Interactive review tests
# ---------------------------------------------------------------------------
class TestReviewFolder:
    """Tests for the _review_folder method."""

    @patch("media_pruner.agent_decision.Prompt.ask")
    def test_keep_action(self, mock_prompt: MagicMock, tmp_path: Path) -> None:
        """Should mark folder as not deleted when user chooses Keep."""
        folder = tmp_path / "photos"
        folder.mkdir()
        report = make_report(folder, score=5)

        mock_prompt.return_value = "k"

        agent = DecisionAgent()
        agent._review_folder(report, index=1, total=1)

        assert report.marked_for_delete is False

    @patch("media_pruner.agent_decision.Prompt.ask")
    def test_skip_action(self, mock_prompt: MagicMock, tmp_path: Path) -> None:
        """Should not change marked_for_delete when user chooses Skip."""
        folder = tmp_path / "photos"
        folder.mkdir()
        report = make_report(folder, score=5)

        mock_prompt.return_value = "s"

        agent = DecisionAgent()
        agent._review_folder(report, index=1, total=1)

        # Skip doesn't change the flag (defaults to False)
        assert report.marked_for_delete is False

    @patch("media_pruner.agent_decision.Prompt.ask")
    def test_delete_high_score_triggers_verification(
        self, mock_prompt: MagicMock, tmp_path: Path
    ) -> None:
        """Should ask for confirmation when deleting a high-score folder."""
        folder = tmp_path / "photos"
        folder.mkdir()
        report = make_report(folder, score=9)

        # First prompt: action = "d" (delete)
        # Second prompt: confirmation = "n" (no, cancel)
        mock_prompt.side_effect = ["d", "n"]

        agent = DecisionAgent()
        agent._review_folder(report, index=1, total=1)

        # Should have been prompted twice (action + confirmation)
        assert mock_prompt.call_count == 2
        assert report.marked_for_delete is False  # User cancelled

    @patch("media_pruner.agent_decision.Prompt.ask")
    @patch("media_pruner.agent_decision.DecisionAgent._move_to_trash")
    def test_delete_confirmed_trashes(
        self,
        mock_trash: MagicMock,
        mock_prompt: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Should trash the folder when user confirms deletion."""
        folder = tmp_path / "photos"
        folder.mkdir()
        report = make_report(folder, score=3)  # Low score, no verification

        mock_prompt.return_value = "d"

        agent = DecisionAgent()
        agent._review_folder(report, index=1, total=1)

        mock_trash.assert_called_once_with(report)
        assert report.marked_for_delete is True

    @patch("media_pruner.agent_decision.Prompt.ask")
    def test_delete_high_score_confirmed(
        self, mock_prompt: MagicMock, tmp_path: Path
    ) -> None:
        """Should trash after user confirms high-score deletion."""
        folder = tmp_path / "photos"
        folder.mkdir()
        report = make_report(folder, score=9)

        mock_prompt.side_effect = ["d", "y"]  # Delete + confirm yes

        with patch("media_pruner.agent_decision.DecisionAgent._move_to_trash"):
            agent = DecisionAgent()
            agent._review_folder(report, index=1, total=1)

        assert report.marked_for_delete is True


# ---------------------------------------------------------------------------
# Trash operation tests
# ---------------------------------------------------------------------------
class TestMoveToTrash:
    """Tests for the _move_to_trash method."""

    @patch("media_pruner.agent_decision.subprocess.run")
    def test_moves_files_to_trash(
        self, mock_run: MagicMock, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        """Should call osascript for each media file."""
        folder = tmp_path / "photos"
        folder.mkdir()
        img = folder / "test.jpg"
        img.write_bytes(b"test data")

        report = FolderReport(
            path=folder,
            media_files=[img],
            picture_count=1,
        )

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        agent = DecisionAgent()
        agent._move_to_trash(report)

        # Should have called osascript once per file
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0][0][0] == "osascript"

    @patch("media_pruner.agent_decision.subprocess.run")
    def test_handles_empty_folder(
        self, mock_run: MagicMock, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        """Should handle a folder with no media files."""
        folder = tmp_path / "empty"
        folder.mkdir()

        report = FolderReport(
            path=folder,
            media_files=[],
        )

        agent = DecisionAgent()
        agent._move_to_trash(report)

        # Should not call osascript for empty folder
        mock_run.assert_not_called()

    def test_dry_run_mode(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        """Should not actually trash files in dry run mode."""
        folder = tmp_path / "photos"
        folder.mkdir()
        img = folder / "test.jpg"
        img.write_bytes(b"test")

        report = FolderReport(
            path=folder,
            media_files=[img],
            picture_count=1,
        )

        agent = DecisionAgent(dry_run=True)
        agent._move_to_trash(report)

        captured = capsys.readouterr()
        assert "Would trash" in captured.out


# ---------------------------------------------------------------------------
# Open in Finder tests
# ---------------------------------------------------------------------------
class TestOpenInFinder:
    """Tests for the _open_in_finder method."""

    @patch("media_pruner.agent_decision.subprocess.run")
    def test_opens_folder(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Should call 'open' command with the folder path."""
        folder = tmp_path / "photos"
        folder.mkdir()

        agent = DecisionAgent()
        agent._open_in_finder(folder)

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "open"
        assert call_args[1] == str(folder)


# ---------------------------------------------------------------------------
# Actions summary tests
# ---------------------------------------------------------------------------
class TestActionsSummary:
    """Tests for the _display_actions_summary method."""

    def test_displays_correct_counts(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        """Should display correct kept/trashed/skipped counts."""
        folder1 = tmp_path / "keep_me"
        folder1.mkdir()
        folder2 = tmp_path / "trash_me"
        folder2.mkdir()
        folder3 = tmp_path / "skip_me"
        folder3.mkdir()

        reports = [
            make_report(folder1, score=8),
            make_report(folder2, score=2),
            make_report(folder3, score=5),
        ]
        reports[1].marked_for_delete = True  # Simulate user trashed this one

        agent = DecisionAgent()
        agent._display_actions_summary(reports)

        captured = capsys.readouterr()
        assert "Kept:" in captured.out
        assert "Trashed:" in captured.out


# ---------------------------------------------------------------------------
# Full execute tests
# ---------------------------------------------------------------------------
class TestExecute:
    """Tests for the main execute() method."""

    def test_returns_empty_for_no_reports(self, tmp_path: Path) -> None:
        """Should handle empty reports list."""
        agent = DecisionAgent()
        results = agent.execute([])
        assert results == []

    @patch("media_pruner.agent_decision.Prompt.ask")
    def test_processes_all_folders(
        self, mock_prompt: MagicMock, tmp_path: Path
    ) -> None:
        """Should process all folders in the reports list."""
        folders = []
        for name in ["a", "b", "c"]:
            f = tmp_path / name
            f.mkdir()
            folders.append(f)

        reports = [make_report(f, score=5) for f in folders]
        mock_prompt.return_value = "k"  # Keep all

        agent = DecisionAgent()
        results = agent.execute(reports)

        assert len(results) == 3
        assert all(not r.marked_for_delete for r in results)
