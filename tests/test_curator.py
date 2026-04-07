"""Unit tests for the CuratorAgent and JSON extraction utilities.

Tests cover:
    - JSON extraction from various LLM output formats
    - Retry loop behavior (fail then succeed, always fail)
    - Context-aware prompt building
    - Full analysis pipeline with mocked Ollama
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from media_pruner.agent_curator import CuratorAgent
from media_pruner.models import AnalysisResult, FolderReport
from media_pruner.utils import extract_json, parse_json_with_retry


def make_fake_jpeg(path: Path) -> None:
    """Create a minimal valid JPEG file that PIL can open."""
    from PIL import Image

    img = Image.new("RGB", (100, 100), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    path.write_bytes(buf.getvalue())


# ---------------------------------------------------------------------------
# JSON extraction tests
# ---------------------------------------------------------------------------
class TestExtractJson:
    """Tests for the regex-based JSON extractor."""

    def test_extracts_bare_json(self) -> None:
        """Should extract a bare JSON object."""
        text = '{"scene_type": "landscape", "score": 8}'
        result = extract_json(text)
        assert result == text

    def test_extracts_from_markdown_block(self) -> None:
        """Should extract JSON from a ```json code block."""
        text = '```json\n{"scene_type": "portrait"}\n```'
        result = extract_json(text)
        assert result == '{"scene_type": "portrait"}'

    def test_extracts_from_markdown_without_language(self) -> None:
        """Should extract JSON from a ``` code block without language tag."""
        text = '```\n{"score": 5}\n```'
        result = extract_json(text)
        assert result == '{"score": 5}'

    def test_extracts_with_chatter_before(self) -> None:
        """Should extract JSON when there's text before it."""
        text = 'Here is the analysis:\n\n{"scene_type": "event"}'
        result = extract_json(text)
        assert '"event"' in result

    def test_extracts_with_chatter_after(self) -> None:
        """Should extract JSON when there's text after it."""
        text = '{"scene_type": "street"}\n\nHope this helps!'
        result = extract_json(text)
        assert '"street"' in result

    def test_extracts_nested_json(self) -> None:
        """Should handle JSON with nested objects."""
        text = '{"scene_type": "portrait", "details": {"people": 3}}'
        result = extract_json(text)
        parsed = json.loads(result)
        assert parsed["details"]["people"] == 3

    def test_raises_on_no_json(self) -> None:
        """Should raise ValueError when no JSON is found."""
        text = "This is just plain text with no JSON at all."
        with pytest.raises(ValueError, match="No JSON object found"):
            extract_json(text)

    def test_raises_on_unbalanced_braces(self) -> None:
        """Should raise ValueError on unbalanced braces."""
        text = '{"scene_type": "landscape"'  # Missing closing brace
        with pytest.raises(ValueError):
            extract_json(text)


class TestParseJsonWithRetry:
    """Tests for the parse_json_with_retry wrapper."""

    def test_parses_valid_json(self) -> None:
        """Should parse valid JSON successfully."""
        text = '{"scene_type": "landscape", "score": 8}'
        result = parse_json_with_retry(text)
        assert result["scene_type"] == "landscape"
        assert result["score"] == 8

    def test_raises_on_invalid_json(self) -> None:
        """Should raise ValueError on invalid JSON."""
        text = '{"scene_type": landscape}'  # Missing quotes
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_json_with_retry(text)

    def test_handles_markdown_wrapped_json(self) -> None:
        """Should handle JSON wrapped in markdown code blocks."""
        text = '```json\n{"scene_type": "interior"}\n```'
        result = parse_json_with_retry(text)
        assert result["scene_type"] == "interior"


# ---------------------------------------------------------------------------
# CuratorAgent tests
# ---------------------------------------------------------------------------
class TestCuratorAgentExecute:
    """Tests for the CuratorAgent.execute() method."""

    def test_analyzes_single_folder(
        self, mock_ollama_client: MagicMock, tmp_path: Path
    ) -> None:
        """Should analyze a single folder and attach AnalysisResult."""
        folder = tmp_path / "photos"
        folder.mkdir()
        img = folder / "test.jpg"
        make_fake_jpeg(img)

        report = FolderReport(
            path=folder,
            media_files=[img],
            picture_count=1,
            representative_images=[img],
        )

        client = mock_ollama_client()
        agent = CuratorAgent(ollama_client=client)
        results = agent.execute([report])

        assert len(results) == 1
        assert results[0].analysis is not None
        assert results[0].analysis.scene_type == "landscape"
        assert results[0].analysis.score == 8

    def test_skips_folder_without_images(
        self, mock_ollama_client: MagicMock, tmp_path: Path
    ) -> None:
        """Should skip folders with no representative images."""
        folder = tmp_path / "empty"
        folder.mkdir()

        report = FolderReport(
            path=folder,
            media_files=[],
            representative_images=[],
        )

        client = mock_ollama_client()
        agent = CuratorAgent(ollama_client=client)
        results = agent.execute([report])

        assert len(results) == 1
        assert results[0].analysis is None  # No analysis performed
        # Verify Ollama was NOT called
        client.chat.assert_not_called()

    def test_handles_markdown_chatter(
        self, mock_ollama_client_json_chatter: MagicMock, tmp_path: Path
    ) -> None:
        """Should handle LLM responses wrapped in markdown code blocks."""
        folder = tmp_path / "photos"
        folder.mkdir()
        img = folder / "test.jpg"
        make_fake_jpeg(img)

        report = FolderReport(
            path=folder,
            media_files=[img],
            picture_count=1,
            representative_images=[img],
        )

        agent = CuratorAgent(ollama_client=mock_ollama_client_json_chatter)
        results = agent.execute([report])

        assert results[0].analysis is not None
        assert results[0].analysis.scene_type == "landscape"

    def test_retry_loop_recovers(
        self, mock_ollama_client_fail_then_succeed: MagicMock, tmp_path: Path
    ) -> None:
        """Should recover from initial failures via the retry loop."""
        folder = tmp_path / "photos"
        folder.mkdir()
        img = folder / "test.jpg"
        make_fake_jpeg(img)

        report = FolderReport(
            path=folder,
            media_files=[img],
            picture_count=1,
            representative_images=[img],
        )

        agent = CuratorAgent(
            ollama_client=mock_ollama_client_fail_then_succeed,
            max_retries=3,
        )
        results = agent.execute([report])

        assert results[0].analysis is not None
        # Ollama should have been called 3 times (2 failures + 1 success)
        assert mock_ollama_client_fail_then_succeed.chat.call_count == 3

    def test_retry_loop_exhausted(
        self, mock_ollama_client_always_fails: MagicMock, tmp_path: Path
    ) -> None:
        """Should raise error after all retries are exhausted."""
        folder = tmp_path / "photos"
        folder.mkdir()
        img = folder / "test.jpg"
        make_fake_jpeg(img)

        report = FolderReport(
            path=folder,
            media_files=[img],
            picture_count=1,
            representative_images=[img],
        )

        agent = CuratorAgent(
            ollama_client=mock_ollama_client_always_fails,
            max_retries=2,
        )

        # Should not raise — the agent logs the error and continues
        results = agent.execute([report])
        # Analysis should be None because all retries failed
        assert results[0].analysis is None

    def test_context_memory_affects_prompt(
        self, mock_ollama_client: MagicMock, tmp_path: Path
    ) -> None:
        """Should include context memory in the prompt for subsequent folders."""
        folder1 = tmp_path / "folder1"
        folder1.mkdir()
        folder2 = tmp_path / "folder2"
        folder2.mkdir()

        img1 = folder1 / "test.jpg"
        img2 = folder2 / "test.jpg"
        make_fake_jpeg(img1)
        make_fake_jpeg(img2)

        reports = [
            FolderReport(
                path=folder1,
                media_files=[img1],
                picture_count=1,
                representative_images=[img1],
            ),
            FolderReport(
                path=folder2,
                media_files=[img2],
                picture_count=1,
                representative_images=[img2],
            ),
        ]

        client = mock_ollama_client()
        agent = CuratorAgent(ollama_client=client)
        agent.execute(reports)

        # Check that the second call included context from the first
        second_call_args = client.chat.call_args_list[1]
        prompt_text = second_call_args[1]["messages"][0]["content"]
        assert "Context from previous analysis" in prompt_text


# ---------------------------------------------------------------------------
# AnalysisResult dataclass tests
# ---------------------------------------------------------------------------
class TestAnalysisResult:
    """Tests for the AnalysisResult dataclass."""

    def test_default_values(self) -> None:
        """Should have sensible defaults for optional fields."""
        result = AnalysisResult(
            scene_type="landscape",
            score=5,
            summary="A landscape",
        )

        assert result.people_count == 0
        assert result.people_description == ""
        assert result.emotions_detected == ""
        assert result.raw_json == ""

    def test_frozen_instance(self) -> None:
        """Should be immutable (frozen=True)."""
        result = AnalysisResult(
            scene_type="portrait",
            score=7,
            summary="A portrait",
        )

        with pytest.raises((AttributeError, Exception)):  # FrozenInstanceError
            result.score = 10  # type: ignore[misc]
