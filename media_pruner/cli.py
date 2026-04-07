"""CLI entry point: argument parsing and agent orchestration.

This module ties all three agents together:
    1. LibrarianAgent scans folders and builds reports
    2. CuratorAgent analyzes reports with the Vision model
    3. DecisionAgent presents results and handles interactive cleanup

The CLI uses argparse (standard library) for argument parsing because:
    - No extra dependencies to install
    - Well-documented and widely understood
    - Great for learning Python CLI development
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .agent_curator import CuratorAgent
from .agent_decision import DecisionAgent
from .agent_librarian import LibrarianAgent
from .models import FolderReport

# ---------------------------------------------------------------------------
# Default scan paths
# ---------------------------------------------------------------------------
# These are the most common locations where users accumulate media files.
# We use Path.home() to get the user's home directory, which works on
# macOS, Linux, and Windows (though the subfolder names differ).
# ---------------------------------------------------------------------------
DEFAULT_SCAN_PATHS: list[str] = [
    "~/Desktop",
    "~/Downloads",
    "~/Documents",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Using argparse.Namespace gives us an object with attributes for each
    argument (e.g., args.path instead of args['path']). This is more
    readable and provides autocomplete support in IDEs.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).
              Overridden in tests to pass custom arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="shuttersort",
        description=(
            "ShutterSort — AI-powered media folder analyzer and pruner.\n\n"
            "Scans local media folders, analyzes them with a local Vision "
            "Language Model (Ollama), detects duplicates, and provides an "
            "interactive cleanup interface."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  shuttersort                          # Scan default paths\n"
            "  shuttersort --path ~/Photos          # Scan a custom path\n"
            "  shuttersort --path ~/A ~/B           # Scan multiple paths\n"
            "  shuttersort --dry-run                # Preview without deleting\n"
            "  shuttersort --model llava            # Use a different model\n"
        ),
    )

    parser.add_argument(
        "--path",
        "-p",
        nargs="+",
        type=str,
        default=None,
        help=(
            "One or more directories to scan. "
            "Defaults to ~/Desktop, ~/Downloads, ~/Documents."
        ),
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="llama3.2-vision",
        help="Ollama model to use for vision analysis (default: llama3.2-vision).",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show what would be deleted without actually deleting anything.",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable verbose (DEBUG) logging output.",
    )

    parser.add_argument(
        "--no-interactive",
        action="store_true",
        default=False,
        help="Skip interactive review; just show the summary table.",
    )

    return parser.parse_args(argv)


def expand_paths(path_strings: list[str]) -> list[Path]:
    """Expand user home directory (~) and resolve paths.

    Path.expanduser() converts ~/Desktop to /Users/username/Desktop.
    Path.resolve() converts relative paths to absolute paths and
    resolves any symlinks.

    We also filter out paths that don't exist, with a warning.

    Args:
        path_strings: List of path strings (may contain ~).

    Returns:
        List of resolved Path objects that exist.
    """
    resolved: list[Path] = []
    for path_str in path_strings:
        path = Path(path_str).expanduser().resolve()
        if path.exists():
            resolved.append(path)
        else:
            print(f"[yellow]Warning: Path does not exist, skipping: {path}[/]")
    return resolved


def setup_logging(verbose: bool) -> None:
    """Configure the logging system.

    Logging levels:
    - DEBUG: Detailed diagnostic info (file paths, API responses)
    - INFO: High-level progress (scanning, analyzing, reviewing)
    - WARNING: Recoverable issues (missing files, parse failures)
    - ERROR: Serious problems (Ollama down, permission denied)

    Args:
        verbose: If True, set level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run(argv: list[str] | None = None) -> int:
    """Main entry point for ShutterSort.

    This function orchestrates the entire pipeline:
    1. Parse arguments
    2. Set up logging
    3. Expand scan paths
    4. Run LibrarianAgent (scan + build reports)
    5. Run CuratorAgent (AI analysis)
    6. Run DecisionAgent (interactive review)
    7. Exit with appropriate code

    Args:
        argv: Command-line arguments (for testing).

    Returns:
        Exit code: 0 for success, 1 for errors.
    """
    # Step 1: Parse arguments
    args = parse_args(argv)

    # Step 2: Set up logging
    setup_logging(args.verbose)

    # Step 3: Determine scan paths
    if args.path:
        scan_paths = expand_paths(args.path)
    else:
        scan_paths = expand_paths(DEFAULT_SCAN_PATHS)

    if not scan_paths:
        print("[red]Error: No valid scan paths provided.[/]")
        return 1

    print("[bold cyan]ShutterSort v0.1.0[/]")
    print(f"[dim]Scanning: {', '.join(str(p) for p in scan_paths)}[/]")
    print(f"[dim]Model: {args.model}[/]\n")

    try:
        # Step 4: LibrarianAgent — scan and build reports
        librarian = LibrarianAgent(model=args.model)
        reports: list[FolderReport] = librarian.execute(scan_paths)

        if not reports:
            print("[yellow]No media folders found.[/]")
            return 0

        print(f"[green]Found {len(reports)} folder(s) with media files.[/]\n")

        # Step 5: CuratorAgent — AI analysis
        print("[bold cyan]Analyzing folders with AI...[/]")
        curator = CuratorAgent(model=args.model)
        reports = curator.execute(reports)
        print()

        # Step 6: DecisionAgent — interactive review
        decision = DecisionAgent(model=args.model, dry_run=args.dry_run)

        if args.no_interactive:
            # Just show the summary table without interactive prompts
            decision._display_summary_table(reports)
        else:
            reports = decision.execute(reports)

        return 0

    except KeyboardInterrupt:
        print("\n[yellow]Interrupted by user. Exiting.[/]")
        return 130  # Standard exit code for Ctrl+C

    except Exception as exc:
        print(f"\n[red]Error: {exc}[/]")
        logging.exception("Unhandled exception")
        return 1


def main() -> None:
    """CLI entry point wrapper.

    This function calls run() and passes the exit code to sys.exit().
    We separate run() (which returns an int) from main() (which exits)
    so that tests can call run() without terminating the test process.
    """
    sys.exit(run())
