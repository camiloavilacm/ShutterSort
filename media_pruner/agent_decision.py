"""DecisionAgent: Interactive review and cleanup interface.

This agent is the "cleaner" of the system. It:
    - Presents analyzed folder reports in a Rich summary table
    - Provides an interactive loop: [K]eep, [D]elete, [O]pen, [S]kip
    - Implements a verification loop for high-score deletions
    - Moves files to macOS Trash via AppleScript (recoverable)
    - Maintains contextual memory of previous decisions

This is the user-facing layer where all the analysis comes together
into actionable decisions.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from .agent_base import MediaAgent
from .models import FolderReport

logger = logging.getLogger(__name__)

console = Console()


class DecisionAgent(MediaAgent):
    """Interactive review and cleanup of analyzed media folders.

    This agent takes the enriched FolderReport objects (with AnalysisResult
    attached by the CuratorAgent) and presents them to the user for review.
    """

    def __init__(
        self,
        model: str = "moondream",
        max_retries: int = 3,
        ollama_client: Any = None,
        dry_run: bool = False,
    ) -> None:
        """Initialize the DecisionAgent.

        Args:
            model: Inherited from MediaAgent (not used directly).
            max_retries: Inherited from MediaAgent.
            ollama_client: Inherited from MediaAgent.
            dry_run: If True, show what would be deleted without actually deleting.
        """
        super().__init__(
            model=model,
            max_retries=max_retries,
            ollama_client=ollama_client,
        )
        self.dry_run = dry_run

        # Contextual memory: remembers the last folder's scene type
        # to provide better context during the interactive review
        self._last_scene_type: str = ""
        self._last_score: int = 0

    def execute(self, reports: list[FolderReport]) -> list[FolderReport]:
        """Run the interactive review loop for all folder reports.

        Flow:
        1. Display the summary table of all folders
        2. For each folder, show details and prompt for action
        3. Execute the chosen action (keep, delete, open, skip)
        4. Return the updated reports with marked_for_delete flags set

        Args:
            reports: List of analyzed FolderReport objects.

        Returns:
            The same reports list, with marked_for_delete updated.
        """
        if not reports:
            console.print("[yellow]No folders to review.[/]")
            return reports

        # Step 1: Show the summary table
        self._display_summary_table(reports)

        # Step 2: Interactive per-folder review
        console.print("\n" + "=" * 60)
        console.print("[bold cyan]Interactive Review[/]")
        console.print("=" * 60)
        console.print(
            "[dim]Actions: [bold green]K[/]eep  "
            "[bold red]D[/]elete  "
            "[bold yellow]O[/]pen in Finder  "
            "[bold blue]S[/]kip[/dim]\n"
        )

        for i, report in enumerate(reports):
            self._review_folder(report, index=i + 1, total=len(reports))

        # Step 3: Summary of actions
        self._display_actions_summary(reports)

        return reports

    def _display_summary_table(self, reports: list[FolderReport]) -> None:
        """Display a Rich table summarizing all folder analyses.

        The table shows:
        - Score (color-coded: green=high, yellow=mid, red=low)
        - Scene type
        - People count
        - Folder path (truncated if long)
        - Summary (truncated)
        - Folder size (human-readable)
        - Picture/Video percentage

        Args:
            reports: List of analyzed FolderReport objects.
        """
        table = Table(
            title="ShutterSort — Folder Analysis Summary",
            show_header=True,
            header_style="bold magenta",
            show_lines=True,
        )

        table.add_column("#", style="dim", width=3)
        table.add_column("Score", justify="center", width=6)
        table.add_column("Scene", width=10)
        table.add_column("People", justify="center", width=6)
        table.add_column("Folder", width=25)
        table.add_column("Summary", width=35)
        table.add_column("Size", justify="right", width=10)
        table.add_column("Pic%", justify="right", width=6)
        table.add_column("Vid%", justify="right", width=6)
        table.add_column("Dupes", justify="center", width=6)

        for i, report in enumerate(reports, 1):
            analysis = report.analysis

            # Color-code the score
            if analysis:
                score_display = str(analysis.score)
                scene = analysis.scene_type
                # Show all scene types if multiple detected
                if analysis.scene_types and len(analysis.scene_types) > 1:
                    scene = ", ".join(analysis.scene_types)
                people = str(analysis.people_count)
                summary = (
                    analysis.summary[:32] + "..."
                    if len(analysis.summary) > 32
                    else analysis.summary
                )
            else:
                score_display = "[dim]N/A[/]"
                scene_display = "[dim]N/A[/]"
                scene = scene_display
                people = "[dim]—[/]"
                summary = "[dim]No analysis[/]"

            # Show full folder path for display
            folder_display = str(report.path)

            # Duplicate indicator
            dupes = "[red]Yes[/]" if report.duplicate_of else "[dim]No[/]"

            table.add_row(
                str(i),
                score_display,
                scene,
                people,
                folder_display,
                summary,
                report.size_human,
                f"{report.picture_percentage:.0f}%",
                f"{report.video_percentage:.0f}%",
                dupes,
            )

        console.print(table)

    def _review_folder(
        self,
        report: FolderReport,
        index: int,
        total: int,
    ) -> None:
        """Review a single folder and prompt for user action.

        This is the core interactive loop. For each folder:
        1. Show a detail panel with analysis results
        2. Prompt for action (K/D/O/S)
        3. Execute the action
        4. Update contextual memory

        Args:
            report: The folder report to review.
            index: Current folder number (1-based).
            total: Total number of folders.
        """
        analysis = report.analysis

        # Build the detail panel
        if analysis:
            detail = (
                f"[bold]Folder:[/] {report.path}\n"
                f"[bold]Files:[/] {len(report.media_files)} "
                f"({report.picture_count} pics, {report.video_count} vids)\n"
                f"[bold]Size:[/] {report.size_human}\n"
                f"[bold]Scene:[/] {analysis.scene_type}\n"
                f"[bold]Score:[/] {analysis.score}/10\n"
                f"[bold]People:[/] {analysis.people_count}\n"
                f"[bold]Emotions:[/] {analysis.emotions_detected}\n"
                f"[bold]Summary:[/] {analysis.summary}"
            )
            if report.duplicate_of:
                detail += (
                    f"\n\n[bold red]⚠ Contains duplicates of:[/] {report.duplicate_of}"
                )
        else:
            detail = (
                f"[bold]Folder:[/] {report.path}\n"
                f"[bold]Files:[/] {len(report.media_files)}\n"
                f"[bold]Size:[/] {report.size_human}\n"
                f"[dim]No AI analysis available[/]"
            )

        # Update contextual memory
        if analysis:
            self._last_scene_type = analysis.scene_type
            self._last_score = analysis.score

        console.print(Panel(detail, title=f"[{index}/{total}] {report.path.name}"))

        # Prompt for action
        while True:
            choice = Prompt.ask(
                "Action",
                choices=["k", "d", "o", "s"],
                default="k",
                show_choices=True,
                show_default=True,
            ).lower()

            if choice == "k":
                report.marked_for_delete = False
                console.print("[green]✓ Keeping this folder.[/]")
                break

            elif choice == "d":
                # Verification loop for high-score folders
                if analysis and analysis.score >= 8:
                    confirm = Prompt.ask(
                        f"[bold red]⚠ This folder has a score of {analysis.score}/10. "
                        f"Are you sure you want to trash it?[/]",
                        choices=["y", "n"],
                        default="n",
                    )
                    if confirm != "y":
                        console.print("[yellow]Cancelled. Keeping this folder.[/]")
                        report.marked_for_delete = False
                        break

                if self.dry_run:
                    console.print(f"[yellow][DRY RUN] Would trash: {report.path}[/]")
                else:
                    self._move_to_trash(report)
                report.marked_for_delete = True
                break

            elif choice == "o":
                self._open_in_finder(report.path)
                # Loop back to prompt again after opening

            elif choice == "s":
                console.print("[blue]⏭ Skipping this folder.[/]")
                break

    def _move_to_trash(self, report: FolderReport) -> None:
        """Move a folder's media files to macOS Trash using AppleScript.

        Using AppleScript (osascript) instead of shutil.rmtree because:
        - AppleScript moves files to the Trash (recoverable)
        - shutil.rmtree permanently deletes (no recovery)
        - Users expect "delete" in a UI context to mean "move to trash"

        The AppleScript command:
            tell application "Finder" to delete POSIX file "/path/to/file"

        This is equivalent to right-clicking a file and selecting "Move to Trash".

        Args:
            report: The folder report whose media files should be trashed.
        """
        media_files = report.media_files
        if not media_files:
            console.print("[yellow]No media files to trash.[/]")
            return

        trashed_count = 0
        failed_count = 0

        for file_path in media_files:
            if self.dry_run:
                trashed_count += 1
                continue

            try:
                # Escape single quotes in the path for AppleScript
                escaped_path = str(file_path).replace("'", "'\\''")

                # Run AppleScript to move file to Trash
                script = (
                    f'tell application "Finder" to delete POSIX file "{escaped_path}"'
                )

                subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                )

                trashed_count += 1
                logger.debug("Trashed: %s", file_path)

            except subprocess.TimeoutExpired:
                logger.error("Timeout trashing file: %s", file_path)
                failed_count += 1

            except subprocess.CalledProcessError as exc:
                logger.warning("AppleScript failed for %s: %s", file_path, exc.stderr)
                failed_count += 1

            except FileNotFoundError:
                logger.warning("File not found, may already be deleted: %s", file_path)
                failed_count += 1

        action_word = "Would trash" if self.dry_run else "Trashed"
        console.print(
            f"[green]✓ {action_word} {trashed_count} file(s) from {report.path.name}[/]"
        )
        if failed_count > 0:
            console.print(f"[yellow]⚠ {failed_count} file(s) could not be trashed.[/]")

    def _open_in_finder(self, folder_path: Path) -> None:
        """Open a folder in macOS Finder.

        Uses the `open` command which is the macOS equivalent of double-clicking
        a folder in Finder.

        Args:
            folder_path: The folder to open.
        """
        try:
            subprocess.run(
                ["open", str(folder_path)],
                capture_output=True,
                timeout=5,
                check=True,
            )
            console.print(f"[blue]📂 Opened in Finder: {folder_path}[/]")
        except Exception as exc:
            console.print(f"[red]Failed to open in Finder: {exc}[/]")

    def _display_actions_summary(self, reports: list[FolderReport]) -> None:
        """Display a summary of all actions taken during the review.

        This gives the user a final overview of what was kept vs trashed.

        Args:
            reports: List of reports with updated marked_for_delete flags.
        """
        kept = sum(1 for r in reports if not r.marked_for_delete)
        trashed = sum(1 for r in reports if r.marked_for_delete)
        skipped = len(reports) - kept - trashed

        console.print("\n" + "=" * 60)
        console.print("[bold cyan]Review Summary[/]")
        console.print("=" * 60)
        console.print(f"  [green]Kept:[/]     {kept} folder(s)")
        console.print(f"  [red]Trashed:[/]  {trashed} folder(s)")
        if skipped > 0:
            console.print(f"  [blue]Skipped:[/]  {skipped} folder(s)")
        console.print()
