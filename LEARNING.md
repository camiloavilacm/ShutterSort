# LEARNING.md — Understanding ShutterSort's Architecture

This guide explains the Python concepts, design patterns, and architectural decisions used in ShutterSort. It's written for developers who are learning Python and want to understand how a real-world CLI application is structured.

## Table of Contents

1. [Project Architecture](#project-architecture)
2. [Type Hints](#type-hints)
3. [Dataclasses](#dataclasses)
4. [Abstract Base Classes](#abstract-base-classes)
5. [Context Managers](#context-managers)
6. [pathlib vs os.path](#pathlib-vs-ospath)
7. [subprocess and AppleScript](#subprocess-and-applescript)
8. [Rich UI Library](#rich-ui-library)
9. [JSON Parsing and Regex](#json-parsing-and-regex)
10. [Error Handling Patterns](#error-handling-patterns)
11. [Garbage Collection](#garbage-collection)
12. [Testing with pytest](#testing-with-pytest)
13. [CI/CD Pipeline](#cicd-pipeline)

---

## Project Architecture

ShutterSort uses an **agent-based architecture** with three specialized agents:

```
                    ┌─────────────────────────────────────────────┐
                    │              cli.py (Entry Point)            │
                    │  Parses args → Orchestrates agents → Exits   │
                    └──────────────────────┬──────────────────────┘
                                           │
                    ┌──────────────────────▼──────────────────────┐
                    │           LibrarianAgent                     │
                    │  Walks dirs → Finds media → Extracts previews│
                    │  → Detects duplicates → Builds FolderReport  │
                    └──────────────────────┬──────────────────────┘
                                           │  FolderReport[]
                    ┌──────────────────────▼──────────────────────┐
                    │            CuratorAgent                      │
                    │  Loads images → Calls Ollama → Parses JSON   │
                    │  → Attaches AnalysisResult to each report    │
                    └──────────────────────┬──────────────────────┘
                                           │  FolderReport[] (enriched)
                    ┌──────────────────────▼──────────────────────┐
                    │           DecisionAgent                      │
                    │  Displays table → Interactive [K/D/O/S] loop │
                    │  → Moves to Trash via AppleScript            │
                    └─────────────────────────────────────────────┘
```

### Why Agents Instead of Functions?

An agent is a class with a single responsibility and an `execute()` method. This pattern offers advantages over plain functions:

- **State**: Agents can hold configuration (model name, retry count, dry-run flag) as instance attributes
- **Testability**: Each agent can be tested in isolation by mocking its dependencies
- **Extensibility**: New agents can be added without modifying existing ones (Open/Closed Principle)
- **Shared behavior**: The `MediaAgent` base class provides common functionality (Ollama connection, retry loop) to all agents

---

## Type Hints

Type hints are annotations that describe the expected types of function parameters and return values. They were introduced in Python 3.5 (PEP 484) and are now a standard part of professional Python code.

### Why Use Type Hints?

```python
# Without type hints — unclear what types are expected
def process(path, count, data):
    ...

# With type hints — immediately clear
def process(path: Path, count: int, data: list[str]) -> dict[str, int]:
    ...
```

Benefits:
1. **Self-documenting**: The function signature tells you what types to pass
2. **IDE support**: Autocomplete, go-to-definition, and inline error highlighting
3. **Static analysis**: Tools like `mypy` catch type errors before runtime
4. **Refactoring safety**: Renaming a type updates all references

### Type Hint Syntax Used in This Project

```python
from __future__ import annotations  # Enables forward references (see below)
from pathlib import Path
from typing import Any, Literal

# Basic types
def greet(name: str) -> str: ...
def count(items: list[str]) -> int: ...

# Optional values (Python 3.10+ syntax using | instead of Optional)
def find(path: str) -> Path | None: ...

# Union types (Python 3.10+ syntax)
def parse(value: str | int) -> float: ...

# Literal types (restricted string values)
SceneType = Literal["landscape", "portrait", "event", "junk", "other"]

# Any (use sparingly — means "any type is acceptable")
def call_ollama(client: Any) -> dict[str, Any]: ...

# Generic dict/list with specific types
def group(files: list[Path]) -> dict[str, list[Path]]: ...
```

### The `from __future__ import annotations` Import

This import enables **postponed evaluation of annotations** (PEP 563). Without it, type hints are evaluated at function definition time, which causes errors when a class references itself:

```python
# Without __future__ annotations — ERROR: Node not defined yet
class Node:
    def __init__(self, next: Node):  # NameError!
        self.next = next

# With __future__ annotations — works fine
from __future__ import annotations

class Node:
    def __init__(self, next: Node):  # OK — annotation is a string
        self.next = next
```

---

## Dataclasses

Dataclasses (introduced in Python 3.7) automatically generate `__init__`, `__repr__`, `__eq__`, and other special methods based on class attributes.

```python
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class FolderReport:
    path: Path
    media_files: list[Path] = field(default_factory=list)
    total_size_bytes: int = 0
    marked_for_delete: bool = False
```

This replaces the verbose manual approach:

```python
# Without dataclass — 20+ lines of boilerplate
class FolderReport:
    def __init__(self, path, media_files=None, total_size_bytes=0, marked_for_delete=False):
        self.path = path
        self.media_files = media_files if media_files is not None else []
        self.total_size_bytes = total_size_bytes
        self.marked_for_delete = marked_for_delete

    def __repr__(self):
        return f"FolderReport(path={self.path}, ...)"

    def __eq__(self, other):
        if not isinstance(other, FolderReport):
            return NotImplemented
        return (self.path == other.path and
                self.media_files == other.media_files and ...)
```

### `frozen=True`

When `frozen=True` is set, the dataclass becomes immutable:

```python
@dataclass(frozen=True)
class AnalysisResult:
    scene_type: str
    score: int
    summary: str

result = AnalysisResult("landscape", 8, "Beautiful view")
result.score = 10  # FrozenInstanceError! Cannot modify
```

Use `frozen=True` for data that represents a "fact" or "snapshot" that shouldn't change after creation.

### `field(default_factory=list)`

Never use mutable defaults like `list` or `dict` directly:

```python
# WRONG — all instances share the same list!
@dataclass
class Bad:
    items: list = []  # Dangerous!

# CORRECT — each instance gets its own list
@dataclass
class Good:
    items: list = field(default_factory=list)
```

---

## Abstract Base Classes

An Abstract Base Class (ABC) defines a contract that subclasses must follow. It cannot be instantiated directly.

```python
from abc import ABC, abstractmethod

class MediaAgent(ABC):
    @abstractmethod
    def execute(self, *args, **kwargs):
        """Every agent must implement this method."""
        ...
```

### Why Use ABCs?

1. **Enforces a contract**: If a subclass forgets to implement `execute()`, Python raises an error at instantiation time
2. **Shared behavior**: The base class provides common methods (like `call_ollama_with_retry()`) that all agents inherit
3. **Polymorphism**: Code can treat all agents uniformly — call `agent.execute()` without knowing the specific type

### The Template Method Pattern

`MediaAgent` uses the Template Method pattern: the base class defines the algorithm skeleton (the retry loop), and subclasses provide the specific steps:

```python
class MediaAgent(ABC):
    def call_ollama_with_retry(self, prompt, images):
        # This is the "template" — the retry algorithm is the same for all agents
        for attempt in range(self.max_retries):
            response = self.ollama_client.chat(...)
            result = parse_json(response)  # Subclasses may customize parsing
            return result
        raise ValueError("All retries failed")

    @abstractmethod
    def execute(self, *args, **kwargs):
        # Each agent implements its own execute() logic
        ...
```

---

## Context Managers

Context managers ensure resources are properly acquired and released, even if errors occur. They use the `with` statement.

```python
# The file is automatically closed when the block exits, even on error
with open("data.txt", "r") as f:
    content = f.read()
# f.close() is called automatically
```

### How Context Managers Work

A context manager implements two methods:
- `__enter__()`: Called when entering the `with` block. Returns the resource.
- `__exit__()`: Called when exiting the block. Handles cleanup.

```python
class ManagedResource:
    def __enter__(self):
        self.resource = acquire_expensive_resource()
        return self.resource

    def __exit__(self, exc_type, exc_val, exc_tb):
        release_expensive_resource(self.resource)
        # Return False to propagate exceptions, True to suppress them
        return False
```

### Context Managers in ShutterSort

```python
# rawpy uses context managers for file handles
with rawpy.imread(str(arw_path)) as raw:
    rgb = raw.postprocess()
# File handle and native memory are released here

# PIL Image also supports context managers
with Image.open(image_path) as img:
    if img.mode == "RGBA":
        img = img.convert("RGB")
# Image file handle is closed here

# tempfile.TemporaryDirectory for automatic cleanup
with tempfile.TemporaryDirectory() as tmpdir:
    frame_path = Path(tmpdir) / "frame.jpg"
    # ... process frame ...
# Entire temporary directory is deleted here
```

---

## pathlib vs os.path

Python has two modules for file system paths: `os.path` (older) and `pathlib` (modern, Python 3.4+).

### os.path (Old Style)

```python
import os

path = "/Users/name/photos/image.jpg"
dirname = os.path.dirname(path)        # "/Users/name/photos"
basename = os.path.basename(path)      # "image.jpg"
ext = os.path.splitext(path)[1]        # ".jpg"
joined = os.path.join(dirname, "new.jpg")  # "/Users/name/photos/new.jpg"
exists = os.path.exists(path)          # True/False
```

String-based operations are error-prone and platform-dependent.

### pathlib (Modern Style — Used in ShutterSort)

```python
from pathlib import Path

path = Path("/Users/name/photos/image.jpg")
path.parent        # Path("/Users/name/photos")
path.name          # "image.jpg"
path.suffix        # ".jpg"
path.parent / "new.jpg"   # Path("/Users/name/photos/new.jpg")
path.exists()      # True/False
```

Key advantages:
- **Object-oriented**: Paths are objects with methods, not strings
- **Operator overloading**: `/` joins paths naturally
- **Cross-platform**: Handles `/` vs `\` automatically
- **Chainable**: `Path("a").parent / "b"` reads left-to-right

### ShutterSort Examples

```python
# Expand ~ to home directory
Path("~/Desktop").expanduser()  # Path("/Users/camilo/Desktop")

# Get absolute path
Path("relative/path").resolve()  # Path("/absolute/path")

# Walk directory recursively
for entry in root.rglob("*"):
    if entry.suffix.lower() in {".jpg", ".png"}:
        process(entry)

# Read/write files
path.read_text()    # Returns file content as string
path.read_bytes()   # Returns file content as bytes
path.write_text("hello")  # Writes string to file
```

---

## subprocess and AppleScript

The `subprocess` module runs external commands from Python. ShutterSort uses it for two things:

### 1. Moving Files to Trash (AppleScript)

```python
import subprocess

def move_to_trash(file_path: Path) -> None:
    # Escape single quotes for AppleScript
    escaped = str(file_path).replace("'", "'\\''")

    script = f'tell application "Finder" to delete POSIX file "{escaped}"'

    subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,  # Capture stdout and stderr
        text=True,            # Return strings instead of bytes
        timeout=10,           # Kill if it takes longer than 10 seconds
        check=True,           # Raise CalledProcessError on non-zero exit
    )
```

Why AppleScript instead of `shutil.rmtree`?
- **Recoverable**: Files go to Trash, not permanently deleted
- **User expectation**: "Delete" in a UI context means "move to trash"
- **Safety**: Users can restore files from Trash if they change their mind

### 2. Opening Folders in Finder

```python
subprocess.run(
    ["open", str(folder_path)],
    capture_output=True,
    timeout=5,
    check=True,
)
```

### subprocess.run Parameters Explained

| Parameter | Purpose |
|-----------|---------|
| `["command", "arg1", "arg2"]` | Command and arguments as a list (safer than a string) |
| `capture_output=True` | Capture stdout and stderr instead of printing to terminal |
| `text=True` | Return strings instead of bytes |
| `timeout=10` | Kill the process if it runs longer than 10 seconds |
| `check=True` | Raise `CalledProcessError` if the command returns a non-zero exit code |

---

## Rich UI Library

Rich is a Python library for beautiful terminal output. ShutterSort uses it for:

### Tables

```python
from rich.table import Table
from rich.console import Console

console = Console()

table = Table(
    title="Folder Analysis Summary",
    show_header=True,
    header_style="bold magenta",
    show_lines=True,
)

table.add_column("#", style="dim", width=3)
table.add_column("Score", justify="center", width=6)
table.add_column("Scene", width=10)

# Color-coded score
if score >= 8:
    score_display = f"[bold green]{score}/10[/]"
elif score >= 5:
    score_display = f"[yellow]{score}/10[/]"
else:
    score_display = f"[red]{score}/10[/]"

table.add_row("1", score_display, "landscape")
console.print(table)
```

### Panels

```python
from rich.panel import Panel

detail = "[bold]Folder:[/] /path/to/photos\n[bold]Score:[/] 8/10"
console.print(Panel(detail, title="[1/5] vacation"))
```

### Prompts

```python
from rich.prompt import Prompt

choice = Prompt.ask(
    "Action",
    choices=["k", "d", "o", "s"],
    default="k",
)
```

Rich markup uses square brackets for styling:
- `[bold]text[/]` — bold
- `[red]text[/]` — red color
- `[bold green]text[/]` — bold and green
- `[dim]text[/]` — dimmed/faded
- `[/]` — reset all styles

---

## JSON Parsing and Regex

LLMs don't always return clean JSON. They often wrap it in markdown code blocks or add conversational text. ShutterSort uses a regex-based extractor to handle this.

### The Problem

```
# What we want:
{"scene_type": "landscape", "score": 8}

# What the LLM might return:
Here's my analysis of the images:

```json
{"scene_type": "landscape", "score": 8}
```

I hope this helps! Let me know if you need anything else.
```

### The Solution: Regex Extraction

```python
import re

# Match ```json ... ``` blocks OR bare {...} objects
_JSON_PATTERN = re.compile(r"```(?:json)?\s*([\s\S]*?)```|(\{[\s\S]*\})")

def extract_json(text: str) -> str:
    match = _JSON_PATTERN.search(text)
    if match:
        # Group 1 = content inside code blocks, Group 2 = bare JSON
        candidate = match.group(1) or match.group(2)
        if candidate:
            return candidate.strip()
    raise ValueError(f"No JSON object found in text: {text[:200]!r}")
```

Regex breakdown:
- ` ``` ` — literal triple backticks
- `(?:json)?` — optional "json" language tag (non-capturing group)
- `\s*` — optional whitespace
- `([\s\S]*?)` — capture everything (including newlines), non-greedy
- ` ``` ` — closing triple backticks
- `|` — OR
- `(\{[\s\S]*\})` — capture a bare JSON object (starts with {, ends with })

### Fallback: Brace Counting

If the regex fails, we fall back to counting braces:

```python
start = text.find("{")
if start != -1:
    depth = 0
    for i, char in enumerate(text[start:], start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1].strip()
```

This handles nested objects correctly by tracking brace depth.

---

## Error Handling Patterns

ShutterSort uses several error handling patterns:

### 1. Try/Except with Logging

```python
try:
    result = risky_operation()
except SpecificError as exc:
    logger.warning("Operation failed: %s", exc)
    return default_value
```

Catch specific exceptions, not bare `except:`. Log the error and return a safe default.

### 2. Try/Finally for Cleanup

```python
cap = cv2.VideoCapture(str(mp4_path))
try:
    # Process frames...
finally:
    cap.release()  # Always called, even if an exception occurs
```

The `finally` block runs regardless of whether an exception was raised. This ensures resources are always released.

### 3. Retry Loop with Reflection

```python
for attempt in range(1, max_retries + 1):
    try:
        response = call_llm(prompt)
        return parse_json(response)
    except ValueError as exc:
        logger.warning("Attempt %d failed: %s", attempt, exc)
        prompt = f"{prompt}\n\nERROR: {exc}. Please fix and retry."

raise ValueError(f"Failed after {max_retries} attempts")
```

Instead of blindly retrying, we feed the error back to the LLM so it can self-correct.

### 4. Graceful Degradation

```python
def extract_arw_preview(arw_path: Path) -> Path | None:
    try:
        # Try to extract preview...
        return preview_path
    except Exception:
        return None  # Caller handles the None case gracefully
```

Return `None` instead of raising when a non-critical operation fails. The caller can skip the file and continue.

### 5. KeyboardInterrupt Handling

```python
try:
    run_pipeline()
except KeyboardInterrupt:
    print("Interrupted by user. Exiting.")
    return 130  # Standard exit code for Ctrl+C
```

Allow users to cleanly exit with Ctrl+C at any point.

---

## Garbage Collection

Python uses automatic reference counting for memory management, but circular references require the garbage collector (`gc` module) to clean up.

### Why `gc.collect()` After ARW Processing?

`rawpy` wraps C libraries (LibRaw) that allocate native memory outside Python's heap. Even after the Python object is destroyed, the C library may hold onto memory.

```python
import gc

with rawpy.imread(str(arw_path)) as raw:
    rgb = raw.postprocess()

# Python's reference counter frees the rawpy object,
# but LibRaw's native memory may still be allocated.
gc.collect()  # Force garbage collection to release native memory
```

This is especially important on 16GB machines processing large RAW files, where memory can accumulate quickly.

### When to Use `gc.collect()`

- After processing large binary data (images, videos, RAW files)
- After using libraries with C extensions (rawpy, numpy, opencv)
- In long-running loops that process many files
- **Not** needed for normal Python code — the GC runs automatically

---

## Testing with pytest

### Test Structure

```
tests/
  conftest.py           # Shared fixtures
  test_librarian.py     # LibrarianAgent unit tests
  test_curator.py       # CuratorAgent unit tests
  test_decision.py      # DecisionAgent unit tests
  test_functional.py    # End-to-end pipeline tests
```

### Fixtures

Fixtures are reusable setup code defined in `conftest.py`:

```python
@pytest.fixture
def temp_media_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "test_photos"
    folder.mkdir()
    (folder / "photo.jpg").write_bytes(b"fake data")
    return folder
```

Pytest automatically injects fixtures into test functions by name:

```python
def test_finds_files(temp_media_folder: Path) -> None:
    # temp_media_folder is provided by the fixture
    files = list(temp_media_folder.glob("*.jpg"))
    assert len(files) == 1
```

### Mocking

Use `unittest.mock` to replace external dependencies:

```python
from unittest.mock import MagicMock, patch

@patch("media_pruner.agent_librarian.rawpy")
def test_arw_extraction(mock_rawpy: MagicMock) -> None:
    mock_rawpy.imread.return_value = MagicMock()
    # Test code runs with the mock instead of real rawpy
```

### Running Tests

```bash
# All tests except integration
pytest -m "not integration"

# With coverage
pytest --cov=media_pruner

# Verbose output
pytest -v

# Stop on first failure
pytest -x
```

### Test Markers

```python
@pytest.mark.integration
def test_real_ollama_call() -> None:
    """Requires a running Ollama instance."""
    ...
```

Skip integration tests: `pytest -m "not integration"`

---

## CI/CD Pipeline

### Continuous Integration (CI)

Every pull request triggers automated checks via GitHub Actions:

```yaml
# .github/workflows/ci.yml
jobs:
  lint-and-test:
    runs-on: macos-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e ".[dev]"
      - run: ruff check media_pruner/ tests/
      - run: ruff format --check media_pruner/ tests/
      - run: mypy media_pruner/
      - run: pytest -m "not integration"
```

### Continuous Deployment (CD)

**Staging** (TestPyPI): Auto-publish on merge to `develop`
**Production** (PyPI): Auto-publish on merge to `main`

### Branch Protection

Configure in GitHub Settings → Branches → Branch protection rules:
- Require pull request reviews (1 approval)
- Require status checks to pass (CI)
- Include administrators

---

## Key Python Concepts Summary

| Concept | Where Used | Why |
|---------|-----------|-----|
| Type hints | Every function | Clarity, IDE support, mypy checking |
| Dataclasses | `models.py` | Less boilerplate, immutable data |
| ABC + @abstractmethod | `agent_base.py` | Enforce agent contract |
| Context managers | File I/O, rawpy, PIL | Automatic resource cleanup |
| pathlib | Everywhere | Modern, object-oriented paths |
| subprocess | Trash, Finder | Run macOS system commands |
| Rich | `agent_decision.py` | Beautiful terminal UI |
| Regex | `utils.py` | Extract JSON from LLM chatter |
| gc.collect() | ARW processing | Release native C memory |
| pytest fixtures | `conftest.py` | Reusable test setup |
| Mocking | All unit tests | Isolate from external services |
