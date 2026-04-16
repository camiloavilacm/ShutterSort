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
from datetime import datetime
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
            "  shuttersort --model moondream         # Use a different model\n"
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
        default="moondream",
        help="Ollama model to use for vision analysis (default: moondream).",
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

    parser.add_argument(
        "--export-html",
        type=str,
        default=None,
        help="Export report to HTML file at the specified path.",
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


def export_html_report(reports: list[FolderReport], output_path: Path) -> None:
    """Export folder analysis reports to an HTML file.

    Args:
        reports: List of analyzed FolderReport objects.
        output_path: Path where the HTML file will be saved.
    """
    html_content = _generate_html(reports)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
    print(f"[green]HTML report exported to: {output_path}[/]")


def _generate_html(reports: list[FolderReport]) -> str:
    """Generate HTML content from folder reports.

    Args:
        reports: List of FolderReport objects.

    Returns:
        HTML string content.
    """
    rows = []
    for i, report in enumerate(reports, 1):
        analysis = report.analysis

        if analysis:
            scene_display = analysis.scene_type
            score_display = str(analysis.score)
            summary = analysis.summary
            people_display = str(analysis.people_count)
        else:
            scene_display = "N/A"
            score_display = "N/A"
            summary = "No analysis"
            people_display = "—"

        dupes = "Yes" if report.duplicate_of else "No"

        rows.append(
            f"""
            <tr>
                <td>{i}</td>
                <td>{score_display}</td>
                <td>{scene_display}</td>
                <td>{people_display}</td>
                <td>{report.path}</td>
                <td>{summary}</td>
                <td>{report.size_human}</td>
                <td>{report.picture_percentage:.0f}%</td>
                <td>{report.video_percentage:.0f}%</td>
                <td>{dupes}</td>
            </tr>
            """
        )

    rows_html = "\n".join(rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ShutterSort Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #333;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #4a90d9;
            color: white;
        }}
        tr:hover {{
            background: #f9f9f9;
        }}
        .summary {{
            margin-bottom: 20px;
            padding: 15px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <h1>ShutterSort Report</h1>
    <div class="summary">
        <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><strong>Total Folders:</strong> {len(reports)}</p>
    </div>
    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>Score</th>
                <th>Scene</th>
                <th>People</th>
                <th>Folder</th>
                <th>Summary</th>
                <th>Size</th>
                <th>Pic%</th>
                <th>Vid%</th>
                <th>Dupes</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
</body>
</html>
"""


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

        # Step 5.5: Export HTML report if requested
        if args.export_html:
            output_path = Path(args.export_html).expanduser().resolve()
            export_html_report(reports, output_path)

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
