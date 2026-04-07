"""Shared utilities: JSON extraction, file hashing, macOS filtering.

This module contains pure functions that don't belong to any specific agent.
Keeping utilities separate makes them easier to test in isolation.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Regex-based JSON extractor
# ---------------------------------------------------------------------------
# LLMs often wrap JSON in markdown code blocks or add conversational text.
# This extractor handles all common patterns:
#   1. ```json { ... } ```  (markdown code block)
#   2. { ... }              (bare JSON)
#   3. Text before { ... }  (chatter before JSON)
# ---------------------------------------------------------------------------
_JSON_PATTERN = re.compile(r"```(?:json)?\s*([\s\S]*?)```|(\{[\s\S]*\})")


def extract_json(text: str) -> str:
    """Extract the first valid JSON object from potentially noisy LLM output.

    This function uses a two-step approach:
    1. Try to match markdown code blocks or bare JSON with regex
    2. If found, validate it's parseable JSON

    Why regex first? Because it's much faster than trying to parse the
    entire text as JSON and catching exceptions. We use regex to narrow
    down to candidate blocks, then json.loads() to validate.

    Args:
        text: Raw string from the LLM, possibly containing markdown or chatter.

    Returns:
        Clean JSON string ready for json.loads().

    Raises:
        ValueError: If no JSON object can be found in the text.
    """
    match = _JSON_PATTERN.search(text)
    if match:
        # Group 1 is content inside code blocks, Group 2 is bare JSON
        candidate = match.group(1) or match.group(2)
        if candidate:
            return candidate.strip()

    # Fallback: try to find any balanced JSON object
    # This handles edge cases where the regex above misses
    start = text.find("{")
    if start != -1:
        # Find the matching closing brace by counting depth
        depth = 0
        for i, char in enumerate(text[start:], start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1].strip()

    raise ValueError(f"No JSON object found in text: {text[:200]!r}")


def parse_json_with_retry(text: str) -> dict[str, Any]:
    """Parse JSON from LLM output, raising a descriptive error on failure.

    This is a wrapper around extract_json + json.loads that provides
    clear error messages for the retry loop.

    Args:
        text: Raw string from the LLM.

    Returns:
        Parsed dictionary.

    Raises:
        ValueError: With a detailed message about what went wrong.
    """
    import json  # Local import to avoid circular dependency issues

    json_str = extract_json(text)
    try:
        result: dict[str, Any] = json.loads(json_str)
        return result
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON: {e}. Extracted block: {json_str[:300]!r}"
        ) from e


# ---------------------------------------------------------------------------
# File hashing for duplicate detection
# ---------------------------------------------------------------------------
# We hash the first 1MB of a file + its size. This is a trade-off:
# - Hashing the entire file is more accurate but slow for large RAW files
# - Hashing just the first 1MB is fast and catches most duplicates
# - Adding file size to the key eliminates false positives from files
#   that happen to share the same first 1MB but differ later
# ---------------------------------------------------------------------------
_HASH_CHUNK_SIZE = 1024 * 1024  # 1 MB


def compute_file_signature(file_path: Path) -> str:
    """Compute a unique signature for duplicate detection.

    The signature is: md5(first 1MB) + "_" + file_size

    Using MD5 here is fine because:
    - We're not using it for security
    - MD5 is fast, which matters when scanning thousands of files
    - Collisions are further mitigated by including file size

    Args:
        file_path: Path to the file.

    Returns:
        A string like "a1b2c3d4..._15728640".
    """
    file_size = file_path.stat().st_size

    with open(file_path, "rb") as f:
        chunk = f.read(_HASH_CHUNK_SIZE)
        file_hash = hashlib.md5(chunk).hexdigest()

    return f"{file_hash}_{file_size}"


# ---------------------------------------------------------------------------
# macOS file system hygiene
# ---------------------------------------------------------------------------
# macOS creates hidden metadata files (._filename) and __MACOSX directories
# when copying files from network shares or zip archives. These are not
# real media files and should be excluded from analysis.
# ---------------------------------------------------------------------------
def is_macos_metadata(name: str) -> bool:
    """Check if a file/directory name is macOS metadata.

    Args:
        name: The filename or directory name (not the full path).

    Returns:
        True if this is a macOS metadata artifact.
    """
    return name.startswith("._") or name == "__MACOSX"


def should_include_path(path: Path) -> bool:
    """Determine if a path should be included in the scan.

    This filters out:
    - macOS metadata files (._*)
    - __MACOSX directories (and any files inside them)
    - Hidden files/directories (starting with .)
    - Non-media files
    """
    name = path.name

    # Skip macOS metadata files
    if is_macos_metadata(name):
        return False

    # Skip files inside __MACOSX directories
    if "__MACOSX" in path.parts:
        return False

    # Skip hidden files
    if name.startswith(".") and path.is_file():
        return False

    # Only include media files
    if path.is_file():
        return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".arw", ".mp4"}

    return True


# ---------------------------------------------------------------------------
# Supported media extensions
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".arw"})
VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4"})
ALL_MEDIA_EXTENSIONS: frozenset[str] = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
