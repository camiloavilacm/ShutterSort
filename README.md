# ShutterSort

AI-powered media folder analyzer and pruner. Scan your photo libraries, get intelligent scene analysis from a local Vision model, detect duplicates, and clean up with confidence.

```
pip install shuttersort
shuttersort
```

## What It Does

| Feature | Description |
|---------|-------------|
| **AI Scene Analysis** | Classifies folders as landscape, portrait, event, junk, etc. using `llama3.2-vision` |
| **Quality Scoring** | Rates each folder 1-10 based on composition, lighting, and content value |
| **People Detection** | Counts people and describes appearances, emotions, and context |
| **Duplicate Detection** | Finds duplicate files across folders using content hashing (first 1MB + file size) |
| **Interactive Cleanup** | Review each folder in a Rich table, then Keep, Delete (to Trash), Open, or Skip |
| **RAW Support** | Extracts previews from Sony ARW files using `rawpy` |
| **Video Support** | Extracts 3 representative frames from MP4s using OpenCV |
| **100% Local** | All AI runs locally via Ollama — no cloud, no uploads, no API keys |

## Quick Install

### Prerequisites

1. **Python 3.10+**
   ```bash
   python3 --version  # Must be 3.10 or higher
   ```

2. **Ollama** installed and running
   ```bash
   # Install Ollama (macOS)
   brew install ollama

   # Start the Ollama service
   ollama serve

   # Pull the vision model (required)
   ollama pull llama3.2-vision
   ```

3. **macOS Full Disk Access** (required for scanning Desktop, Downloads, Documents)
   - Open **System Settings** → **Privacy & Security** → **Full Disk Access**
   - Click the **+** button and add your terminal app:
     - **Terminal.app**: `/System/Applications/Utilities/Terminal.app`
     - **iTerm2**: `/Applications/iTerm.app`
     - **VS Code Terminal**: `/Applications/Visual Studio Code.app`
   - Restart your terminal after granting access

   Without Full Disk Access, macOS will silently return empty results when scanning protected folders.

### Install ShutterSort

```bash
pip install shuttersort
```

Or from source:

```bash
git clone https://github.com/camiloavilacm/ShutterSort.git
cd ShutterSort
pip install -e .
```

## Usage

### Basic Scan (Default Paths)

Scans `~/Desktop`, `~/Downloads`, and `~/Documents`:

```bash
shuttersort
```

### Custom Paths

```bash
# Single path
shuttersort --path ~/Photos

# Multiple paths
shuttersort --path ~/Photos ~/Pictures ~/ExternalDrive

# Shorthand
shuttersort -p ~/Photos
```

### Different Model

```bash
shuttersort --model llava
shuttersort -m llava
```

### Dry Run (Preview Only)

See what would be deleted without actually deleting anything:

```bash
shuttersort --dry-run
```

### Verbose Output

Show detailed debug logging:

```bash
shuttersort --verbose
shuttersort -v
```

### Non-Interactive Mode

Just show the summary table without the interactive review prompts:

```bash
shuttersort --no-interactive
```

## How It Works

ShutterSort uses a **three-agent architecture**:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  LibrarianAgent │────>│  CuratorAgent   │────>│ DecisionAgent   │
│                 │     │                 │     │                 │
│ • Walks folders │     │ • Calls Ollama  │     │ • Rich table    │
│ • Finds media   │     │ • Analyzes imgs │     │ • [K/D/O/S] loop│
│ • Extracts ARW  │     │ • Scores 1-10   │     │ • AppleScript   │
│ • Finds dupes   │     │ • Detects people│     │ • Trash to Finder│
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

1. **LibrarianAgent** walks your folders, finds all media files (JPG, PNG, ARW, MP4), extracts previews from RAW files, and detects duplicates.
2. **CuratorAgent** sends up to 5 representative images per folder to `llama3.2-vision` and returns a structured analysis (scene type, score, people count, emotions).
3. **DecisionAgent** presents everything in a color-coded Rich table and walks you through each folder with an interactive `[K]eep / [D]elete / [O]pen / [S]kip` loop.

### Duplicate Detection

Files are matched by a composite key: **MD5 hash of the first 1MB + file size**. When duplicates are found across folders, ShutterSort suggests keeping the copy in the folder with the higher AI score.

### Delete Behavior (Trash vs Permanent)

When you choose **Delete**, ShutterSort uses **AppleScript** to move files to the macOS Trash:

```applescript
tell application "Finder" to delete POSIX file "/path/to/file"
```

This is equivalent to right-clicking a file and selecting "Move to Trash." Files can be recovered from the Trash until you empty it. ShutterSort **never** permanently deletes files.

### Temporary File Handling

All extracted previews (from ARW files) and video frames (from MP4s) are saved to temporary files using Python's `tempfile` module. These are cleaned up automatically after analysis. The `gc.collect()` call after ARW processing ensures native C memory from `rawpy` is released promptly, keeping RAM usage low on 16GB machines.

## Output Example

```
ShutterSort — Folder Analysis Summary
┏━━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
┃ # ┃ Score ┃ Scene     ┃ People ┃ Folder           ┃ Summary                           ┃ Size     ┃ Pic%   ┃ Vid%   ┃ Dupes ┃
┡━━━╇━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
│ 1 │ 8/10  │ landscape │ 2      │ .../vacation     │ Beautiful beach photos from vac…  │ 245.30 MB│ 100%   │ 0%     │ No    │
│ 2 │ 2/10  │ junk      │ 0      │ .../screenshots  │ Screenshots of documents and re…  │ 12.50 MB │ 100%   │ 0%     │ Yes   │
│ 3 │ 9/10  │ event     │ 6      │ .../family       │ Birthday party with family memb…  │ 512.00 MB│ 80%    │ 20%    │ No    │
└───┴───────┴───────────┴────────┴──────────────────┴───────────────────────────────────┴──────────┴────────┴────────┴───────┘
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **"No media folders found"** | Check Full Disk Access for your terminal app (see Prerequisites above) |
| **Ollama connection refused** | Run `ollama serve` in another terminal tab |
| **Model not found** | Run `ollama pull llama3.2-vision` |
| **ARW files fail to process** | Ensure `rawpy` is installed: `pip install rawpy` |
| **Slow analysis** | Large folders take longer; use `--verbose` to see progress |
| **JSON parse errors** | The retry loop handles this automatically (up to 3 retries) |

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

Quick start for contributors:

```bash
git clone https://github.com/camiloavilacm/ShutterSort.git
cd ShutterSort
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -m "not integration"
```

### Branching Model

- `main` — Production (auto-publishes to PyPI)
- `develop` — Staging (auto-publishes to TestPyPI)
- `feature/*` — Feature branches (PR → develop)

### CI/CD

Every PR runs lint (ruff), type checks (mypy), and tests (pytest) on Python 3.10–3.13.

## License

MIT — see [LICENSE](LICENSE) for details.
