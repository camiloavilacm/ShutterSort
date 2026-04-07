"""LibrarianAgent: File system manager and evidence preparer.

This agent is responsible for:
    - Walking directories and finding media files
    - Filtering macOS metadata (._files, __MACOSX dirs)
    - Extracting previews from ARW (RAW) files using rawpy
    - Extracting frames from MP4 videos using OpenCV
    - Computing file hashes for duplicate detection
    - Calculating folder sizes and media statistics
    - Selecting representative images for the CuratorAgent

Think of this agent as a librarian who catalogs all the books (media files),
creates summaries (previews/frames), and prepares the evidence for the judge.
"""

from __future__ import annotations

import gc
import logging
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from .agent_base import MediaAgent
from .models import FolderReport
from .utils import (
    ALL_MEDIA_EXTENSIONS,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    compute_file_signature,
    should_include_path,
)

logger = logging.getLogger(__name__)


class LibrarianAgent(MediaAgent):
    """Manages file system operations and prepares media for analysis.

    This agent does NOT call the Vision model. Its job is purely:
    1. Find all media files in the given paths
    2. Group them by folder
    3. Extract previews/frames as needed
    4. Detect duplicates
    5. Prepare FolderReport objects for the CuratorAgent
    """

    def __init__(
        self,
        model: str = "llama3.2-vision",
        max_retries: int = 3,
        ollama_client: Any = None,
        max_image_size: int = 1024,
    ) -> None:
        """Initialize the LibrarianAgent.

        Args:
            model: Inherited from MediaAgent (not used directly by Librarian).
            max_retries: Inherited from MediaAgent.
            ollama_client: Inherited from MediaAgent.
            max_image_size: Maximum dimension for extracted previews (pixels).
        """
        super().__init__(
            model=model,
            max_retries=max_retries,
            ollama_client=ollama_client,
        )
        self.max_image_size = max_image_size

        # Track all file signatures for duplicate detection across folders
        # Key: file signature (hash + size), Value: list of (path, size)
        self._file_signatures: dict[str, list[tuple[Path, int]]] = {}

    def execute(self, scan_paths: list[Path]) -> list[FolderReport]:
        """Scan the given paths and produce FolderReport objects.

        This is the main entry point. The flow is:
        1. Walk each path and collect media files grouped by folder
        2. For each folder, calculate statistics (size, counts)
        3. Extract representative images for AI analysis
        4. Detect duplicates across all scanned folders
        5. Return the list of FolderReport objects

        Args:
            scan_paths: List of directories to scan.

        Returns:
            List of FolderReport objects, one per folder containing media.
        """
        logger.info("Scanning %d path(s): %s", len(scan_paths), scan_paths)

        # Step 1: Walk directories and group files by folder
        # We use a dict to automatically merge files from the same folder
        # if multiple scan paths overlap
        folder_files: dict[Path, list[Path]] = {}
        for scan_path in scan_paths:
            self._walk_directory(scan_path, folder_files)

        if not folder_files:
            logger.warning("No media files found in any scan path.")
            return []

        logger.info("Found %d folder(s) with media files.", len(folder_files))

        # Step 2: Build FolderReport objects
        reports: list[FolderReport] = []
        for folder_path, media_files in sorted(folder_files.items()):
            report = self._build_folder_report(folder_path, media_files)
            reports.append(report)

        # Step 3: Detect duplicates across all folders
        self._detect_duplicates(reports)

        return reports

    def _walk_directory(
        self,
        root: Path,
        folder_files: dict[Path, list[Path]],
    ) -> None:
        """Recursively walk a directory and collect media files.

        We use pathlib.Path.rglob() instead of os.walk() because:
        - rglob() returns Path objects directly (no need to join strings)
        - It's more readable and Pythonic
        - It handles path separators correctly on all platforms

        However, rglob() doesn't let us skip directories early, so we
        check each entry with should_include_path() before processing.

        Args:
            root: The root directory to start walking from.
            folder_files: Dict to populate with folder -> files mapping.
        """
        if not root.exists():
            logger.warning("Path does not exist, skipping: %s", root)
            return

        if root.is_file():
            # If a single file is given, treat its parent as the folder
            if root.suffix.lower() in ALL_MEDIA_EXTENSIONS:
                folder_files.setdefault(root.parent, []).append(root)
            return

        # Walk all entries recursively
        for entry in root.rglob("*"):
            # Skip macOS metadata and non-media files
            if not should_include_path(entry):
                continue

            # Group by immediate parent folder (not subdirectories)
            # This means each subdirectory gets its own FolderReport
            parent = entry.parent
            folder_files.setdefault(parent, []).append(entry)

    def _build_folder_report(
        self,
        folder_path: Path,
        media_files: list[Path],
    ) -> FolderReport:
        """Build a FolderReport for a single folder.

        This calculates:
        - Total size of all media files
        - Picture count vs video count
        - Representative images for AI analysis

        Args:
            folder_path: The folder being reported on.
            media_files: List of media file paths in this folder.

        Returns:
            A populated FolderReport object.
        """
        total_size = 0
        picture_count = 0
        video_count = 0

        for file_path in media_files:
            try:
                file_size = file_path.stat().st_size
                total_size += file_size

                if file_path.suffix.lower() in IMAGE_EXTENSIONS:
                    picture_count += 1
                elif file_path.suffix.lower() in VIDEO_EXTENSIONS:
                    video_count += 1
            except OSError as exc:
                logger.warning("Could not stat file %s: %s", file_path, exc)

        # Select representative images for AI analysis
        # Strategy: pick the largest images (they tend to have more detail)
        representative = self._select_representive_images(media_files, max_count=5)

        report = FolderReport(
            path=folder_path,
            media_files=sorted(media_files),
            total_size_bytes=total_size,
            picture_count=picture_count,
            video_count=video_count,
            representative_images=representative,
        )

        logger.debug(
            "Built report for %s: %d files, %s",
            folder_path.name,
            len(media_files),
            report.size_human,
        )

        return report

    def _select_representive_images(
        self,
        media_files: list[Path],
        max_count: int = 5,
    ) -> list[Path]:
        """Select the most representative images for AI analysis.

        Strategy: Sort images by file size (descending) and pick the top N.
        Larger files tend to have more detail and fewer compression artifacts,
        making them better candidates for vision analysis.

        We only select image files (not videos) because:
        - The Vision model works best with still images
        - Video frames are extracted separately by the frame extractor
        - Mixing video files into the image batch would confuse the model

        Args:
            media_files: All media files in the folder.
            max_count: Maximum number of images to select.

        Returns:
            List of up to max_count image paths, sorted by size (largest first).
        """
        # Filter to only image files
        image_files = [f for f in media_files if f.suffix.lower() in IMAGE_EXTENSIONS]

        # Sort by file size (descending) and take the top N
        image_files.sort(
            key=lambda p: p.stat().st_size if p.exists() else 0,
            reverse=True,
        )

        return image_files[:max_count]

    def extract_arw_preview(self, arw_path: Path) -> Path | None:
        """Extract a JPEG preview from a Sony ARW RAW file.

        ARW files contain embedded JPEG previews that we can extract without
        fully decoding the RAW data. This is much faster and uses less memory.

        Memory management:
        - We use tempfile.NamedTemporaryFile() for automatic cleanup
        - We call gc.collect() after processing to free rawpy's C buffers
        - rawpy holds onto native memory that Python's GC doesn't track

        Args:
            arw_path: Path to the ARW file.

        Returns:
            Path to the extracted JPEG preview, or None on failure.
        """
        try:
            import rawpy

            # Use a temporary file for the extracted preview
            # delete=False because we need to access it after the context exits
            # The caller is responsible for cleanup
            with rawpy.imread(str(arw_path)) as raw:
                # Extract the embedded JPEG preview (fast, no demosaicing)
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    half_size=True,  # Half resolution for speed
                    no_auto_bright=True,
                )

            # Resize if larger than max_image_size
            image = Image.fromarray(rgb)
            if max(image.size) > self.max_image_size:
                image.thumbnail(
                    (self.max_image_size, self.max_image_size),
                    Image.Resampling.LANCZOS,
                )

            # Save to a temporary file (delete=False since caller manages cleanup)
            import tempfile

            with tempfile.NamedTemporaryFile(
                suffix=".jpg",
                delete=False,
            ) as tmp:
                image.save(tmp.name, "JPEG", quality=85)

            # Force garbage collection to free rawpy's native memory
            # rawpy uses C libraries that hold onto memory even after
            # the Python object is garbage collected. gc.collect() forces
            # Python to clean up circular references and release memory.
            gc.collect()

            logger.debug(
                "Extracted ARW preview: %s -> %s (%dx%d)",
                arw_path.name,
                tmp.name,
                image.size[0],
                image.size[1],
            )

            return Path(tmp.name)

        except Exception as exc:
            logger.warning(
                "Failed to extract ARW preview from %s: %s",
                arw_path,
                exc,
            )
            gc.collect()  # Still collect on error to free any partial allocations
            return None

    def extract_video_frames(self, mp4_path: Path) -> list[Path]:
        """Extract exactly 3 frames from an MP4 video at 10%, 50%, 90%.

        Using OpenCV to extract frames instead of ffmpeg because:
        - opencv-python-headless is already a dependency
        - It gives us precise frame-by-frame control
        - No subprocess overhead

        The 3-frame strategy captures:
        - 10%: Beginning context (establishing shot)
        - 50%: Middle content (main action)
        - 90%: End context (conclusion)

        Args:
            mp4_path: Path to the MP4 file.

        Returns:
            List of 3 temporary frame file paths, or empty list on failure.
        """
        import cv2

        cap = cv2.VideoCapture(str(mp4_path))
        if not cap.isOpened():
            logger.warning("Could not open video: %s", mp4_path)
            return []

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                logger.warning("Video has 0 frames: %s", mp4_path)
                return []

            # Calculate frame positions at 10%, 50%, 90%
            positions = [max(0, int(total_frames * pct)) for pct in (0.1, 0.5, 0.9)]

            frame_paths: list[Path] = []

            for _idx, frame_pos in enumerate(positions):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()

                if not ret:
                    logger.warning(
                        "Could not read frame %d from %s", frame_pos, mp4_path
                    )
                    continue

                # Resize frame for analysis
                height, width = frame.shape[:2]
                if max(height, width) > self.max_image_size:
                    scale = self.max_image_size / max(height, width)
                    frame = cv2.resize(
                        frame,
                        (int(width * scale), int(height * scale)),
                        interpolation=cv2.INTER_AREA,
                    )

                # Save to temporary file (delete=False since caller manages cleanup)
                with tempfile.NamedTemporaryFile(
                    suffix=".jpg",
                    delete=False,
                ) as tmp:
                    # cv2 uses BGR, convert to RGB for JPEG saving
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    pil_image.save(tmp.name, "JPEG", quality=85)

                frame_paths.append(Path(tmp.name))

            logger.debug("Extracted %d frames from %s", len(frame_paths), mp4_path.name)
            return frame_paths

        finally:
            # Always release the video capture handle
            cap.release()

    def load_image_for_analysis(self, image_path: Path) -> bytes | None:
        """Load and optionally resize an image for AI analysis.

        This handles all image types: JPG, PNG, ARW, and extracted frames.
        For ARW files, it extracts the preview first.

        Args:
            image_path: Path to the image file.

        Returns:
            JPEG-encoded image bytes, or None on failure.
        """
        suffix = image_path.suffix.lower()

        try:
            if suffix == ".arw":
                # Extract preview from RAW file
                preview_path = self.extract_arw_preview(image_path)
                if preview_path is None:
                    return None
                # Load the extracted preview
                with open(preview_path, "rb") as f:
                    data = f.read()
                # Clean up the temporary preview file
                preview_path.unlink(missing_ok=True)
                return data

            elif suffix == ".mp4":
                # This shouldn't happen — frames should be extracted separately
                logger.warning("load_image_for_analysis called on MP4: %s", image_path)
                return None

            else:
                # Regular image (JPG, PNG)
                with Image.open(image_path) as img:
                    # Convert to RGB if necessary (PNGs can have alpha channel)
                    processed: Image.Image = (
                        img.convert("RGB") if img.mode in ("RGBA", "P", "LA") else img
                    )

                    # Resize if too large (reduces API call size)
                    if max(processed.size) > self.max_image_size:
                        processed.thumbnail(
                            (self.max_image_size, self.max_image_size),
                            Image.Resampling.LANCZOS,
                        )

                    # Save to bytes buffer
                    import io

                    buf = io.BytesIO()
                    processed.save(buf, format="JPEG", quality=85)
                    return buf.getvalue()

        except Exception as exc:
            logger.warning("Failed to load image %s for analysis: %s", image_path, exc)
            return None

    def _detect_duplicates(self, reports: list[FolderReport]) -> None:
        """Detect duplicate files across all folder reports.

        Duplicates are identified by matching file signatures (hash + size).
        When duplicates are found, we suggest keeping the copy in the
        highest-scoring folder (once analysis is complete).

        This method populates the `duplicate_of` field on FolderReport objects
        and creates DuplicateGroup entries.

        Args:
            reports: List of FolderReport objects to check for duplicates.
        """
        # Build a mapping of signature -> files across all folders
        signature_map: dict[str, list[tuple[Path, Path]]] = {}
        # Key: signature, Value: list of (file_path, folder_path)

        for report in reports:
            for file_path in report.media_files:
                if not file_path.exists():
                    continue

                try:
                    sig = compute_file_signature(file_path)
                    signature_map.setdefault(sig, []).append((file_path, report.path))
                except OSError as exc:
                    logger.warning(
                        "Could not compute signature for %s: %s",
                        file_path,
                        exc,
                    )

        # Find groups with files in multiple folders (true cross-folder dupes)
        for _sig, file_list in signature_map.items():
            folders_with_file = set(folder for _, folder in file_list)
            if len(folders_with_file) > 1:
                # This file exists in multiple folders — it's a duplicate
                for _file_path, folder_path in file_list:
                    # Mark all copies except the first as duplicates
                    first_folder = file_list[0][1]
                    if folder_path != first_folder:
                        # Find the report for this folder and mark it
                        for report in reports:
                            if report.path == folder_path:
                                report.duplicate_of = first_folder
                                break

        logger.info(
            "Duplicate detection complete. Found duplicates in %d folder(s).",
            sum(1 for r in reports if r.duplicate_of is not None),
        )
