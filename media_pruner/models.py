"""Typed dataclasses that represent analysis results and folder reports.

These dataclasses are the shared language between all three agents.
Using dataclasses instead of plain dicts gives us:
    - Type safety (my catches wrong field types)
    - Auto-generated __init__, __repr__, __eq__
    - Clear contract: every agent knows exactly what fields to expect
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Scene type enumeration
# ---------------------------------------------------------------------------
# We use Literal instead of Enum for simplicity and better JSON serialization.
# The CuratorAgent must return one of these exact strings.
SceneType = Literal[
    "landscape",
    "interior",
    "portrait",
    "street",
    "event",
    "junk",
    "other",
]


# ---------------------------------------------------------------------------
# Analysis result from the CuratorAgent
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AnalysisResult:
    """The structured analysis output from the Vision model.

    frozen=True makes instances immutable, which prevents accidental
    modification after creation. This is a good practice for data that
    represents a "fact" or "snapshot" at a point in time.
    """

    scene_type: SceneType
    """Classification of the dominant scene type in the folder."""

    score: int
    """Quality/relevance score from 1-10."""

    summary: str
    """One-sentence description of the folder contents."""

    scene_types: list[str] = field(default_factory=list)
    """List of all scene types detected in the images."""

    primary_scene: str = ""
    """The primary/dominant scene type."""

    people_count: int = 0
    """Number of people detected in the images."""

    people_description: str = ""
    """Description of people: appearance, age range, clothing, pose."""

    emotions_detected: str = ""
    """Mood/vibe detected (e.g., 'happy, candid, serious')."""

    raw_json: str = ""
    """The raw JSON string returned by the model (for debugging)."""


# ---------------------------------------------------------------------------
# Folder report from the LibrarianAgent
# ---------------------------------------------------------------------------
@dataclass
class FolderReport:
    """A comprehensive report about a single folder's media contents.

    Unlike AnalysisResult, this is mutable (frozen=False) because the
    DecisionAgent may update fields like 'marked_for_delete' during
    the interactive review process.
    """

    path: Path
    """Absolute path to the folder."""

    media_files: list[Path] = field(default_factory=list)
    """List of all media files found (JPG, PNG, ARW, MP4)."""

    total_size_bytes: int = 0
    """Total size of all media files in bytes."""

    picture_count: int = 0
    """Number of image files (JPG, PNG, ARW)."""

    video_count: int = 0
    """Number of video files (MP4)."""

    analysis: AnalysisResult | None = None
    """AI analysis result, populated by CuratorAgent."""

    representative_images: list[Path] = field(default_factory=list)
    """Up to 5 representative images selected for AI analysis."""

    marked_for_delete: bool = False
    """Whether the user has chosen to delete this folder's media."""

    duplicate_of: Path | None = None
    """If this folder contains duplicates, which folder has the originals."""

    @property
    def picture_percentage(self) -> float:
        """Percentage of media files that are pictures.

        Using @property means this is computed on-demand rather than
        stored as a field. This avoids stale data if picture_count changes.
        """
        total = self.picture_count + self.video_count
        if total == 0:
            return 0.0
        return (self.picture_count / total) * 100

    @property
    def video_percentage(self) -> float:
        """Percentage of media files that are videos."""
        total = self.picture_count + self.video_count
        if total == 0:
            return 0.0
        return (self.video_count / total) * 100

    @property
    def size_human(self) -> str:
        """Human-readable folder size (e.g., '1.2 GB')."""
        return _human_size(self.total_size_bytes)


# ---------------------------------------------------------------------------
# Duplicate group
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DuplicateGroup:
    """A group of files that are duplicates of each other.

    Duplicates are identified by matching hash (first 1MB) + file size.
    This is more reliable than filename-only matching.
    """

    hash_key: str
    """The composite key: md5(first_1MB) + file_size."""

    files: list[tuple[Path, int]] = field(default_factory=list)
    """List of (file_path, file_size) tuples that share this hash."""

    suggested_keep: Path | None = None
    """The file to keep (located in the highest-scoring folder)."""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _human_size(nbytes: int) -> str:
    """Convert bytes to a human-readable string.

    Example: 1_500_000 -> '1.43 MB'
    """
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024.0:
            return f"{nbytes:.2f} {unit}"
        nbytes = int(nbytes / 1024)
    return f"{nbytes:.2f} PB"
