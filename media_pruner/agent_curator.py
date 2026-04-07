"""CuratorAgent: Vision model interface and image analysis.

This agent is the "judge" of the system. Its only job is to:
    - Receive representative images from the LibrarianAgent
    - Send them to the Vision model (llama3.2-vision via Ollama)
    - Parse and validate the JSON response
    - Return a typed AnalysisResult

It does NOT care about file paths, deletion, or duplicates.
It only answers: "What is in these images?"
"""

from __future__ import annotations

import logging
from pathlib import Path

from .agent_base import MediaAgent
from .models import FolderReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The system prompt for the Vision model
# ---------------------------------------------------------------------------
# This prompt is carefully crafted to:
# 1. Define the exact JSON schema expected
# 2. Forbid markdown/code blocks (a common LLM habit)
# 3. Provide clear definitions for each scene_type
# 4. Ask for a 1-10 score based on image quality and content value
# ---------------------------------------------------------------------------
CURATOR_PROMPT = """You are an expert photo analyst. Look at these images and provide a detailed analysis in JSON format.

Analyze the overall content of these images and respond with ONLY a valid JSON object (no markdown, no code blocks, no extra text) matching this exact schema:

{
    "scene_type": "landscape | interior | portrait | street | event | junk | other",
    "score": 1,
    "summary": "Brief 1-sentence description",
    "people_count": 0,
    "people_description": "Appearance, age range, clothing, pose",
    "emotions_detected": "Mood/vibe (e.g., happy, candid, serious)"
}

Scene type definitions:
- landscape: Natural scenery, mountains, beaches, skies, outdoor vistas
- interior: Indoor spaces, rooms, architecture, furniture
- portrait: Photos focused on people, headshots, group photos
- street: Urban scenes, city life, candid street photography
- event: Weddings, parties, concerts, celebrations, gatherings
- junk: Screenshots, memes, receipts, documents, low-value images
- other: Anything that doesn't fit the above categories

Score guidelines:
- 1-3: Low quality, blurry, accidental shots, screenshots
- 4-6: Average quality, standard snapshots, moderately interesting
- 7-8: Good composition, interesting subject, well-lit
- 9-10: Professional quality, stunning composition, emotionally powerful

IMPORTANT: Respond with ONLY the JSON object. Do NOT wrap it in ```json code blocks. Do NOT add any text before or after the JSON."""


class CuratorAgent(MediaAgent):
    """Analyzes folder contents using the Vision model.

    This agent takes FolderReport objects from the LibrarianAgent,
    loads the representative images, sends them to Ollama, and
    enriches the reports with AnalysisResult objects.
    """

    def execute(
        self,
        reports: list[FolderReport],
        context_memory: str = "",
    ) -> list[FolderReport]:
        """Analyze all folder reports using the Vision model.

        For each folder:
        1. Load representative images as JPEG bytes
        2. Build a prompt with optional context from previous folders
        3. Call Ollama with retry logic
        4. Attach the AnalysisResult to the FolderReport

        Args:
            reports: List of FolderReport objects from LibrarianAgent.
            context_memory: Optional context from previously analyzed folders
                          (e.g., "Previous folder was 'Wedding Photos - event'").

        Returns:
            The same list of reports, now enriched with analysis results.
        """
        logger.info("CuratorAgent analyzing %d folder(s)...", len(reports))

        analyzed_count = 0
        for i, report in enumerate(reports):
            if not report.representative_images:
                logger.debug(
                    "Skipping folder %s: no representative images",
                    report.path,
                )
                continue

            # Build context-aware prompt
            prompt = self._build_prompt(report, context_memory)

            # Load all representative images as bytes
            image_bytes_list: list[bytes] = []
            for img_path in report.representative_images:
                img_bytes = self._load_image(img_path)
                if img_bytes is not None:
                    image_bytes_list.append(img_bytes)

            if not image_bytes_list:
                logger.warning("Could not load any images for folder %s", report.path)
                continue

            # Call Ollama with retry logic (inherited from MediaAgent)
            try:
                analysis = self.call_ollama_with_retry(
                    prompt=prompt,
                    images=image_bytes_list,
                )
                report.analysis = analysis
                analyzed_count += 1

                logger.info(
                    "[%d/%d] Analyzed %s: scene=%s, score=%d, people=%d",
                    i + 1,
                    len(reports),
                    report.path.name,
                    analysis.scene_type,
                    analysis.score,
                    analysis.people_count,
                )

                # Update context memory for next folder
                context_memory = (
                    f"Previous folder '{report.path.name}' was classified as "
                    f"'{analysis.scene_type}' (score: {analysis.score}/10). "
                    f"Summary: {analysis.summary}"
                )

            except Exception as exc:
                logger.error("Failed to analyze folder %s: %s", report.path, exc)

        logger.info(
            "CuratorAgent complete. Analyzed %d/%d folders.",
            analyzed_count,
            len(reports),
        )

        return reports

    def _build_prompt(
        self,
        report: FolderReport,
        context_memory: str,
    ) -> str:
        """Build the analysis prompt with optional context.

        Context memory helps the model understand relationships between
        folders. For example, if the previous folder was "Wedding Photos",
        the model might better classify the current folder as "event" if
        it contains similar content.

        Args:
            report: The folder report being analyzed.
            context_memory: Context from previously analyzed folders.

        Returns:
            The complete prompt string.
        """
        prompt = CURATOR_PROMPT

        # Add context if available
        if context_memory:
            prompt = f"{prompt}\n\nContext from previous analysis: {context_memory}"

        # Add folder-specific info
        prompt += (
            f"\n\nFolder: {report.path.name}\n"
            f"Files: {len(report.media_files)} total "
            f"({report.picture_count} images, {report.video_count} videos)"
        )

        return prompt

    def _load_image(self, image_path: Path) -> bytes | None:
        """Load an image file as JPEG bytes for Ollama.

        This handles JPG, PNG, and extracted ARW previews.
        MP4 frames are already extracted as JPG by the LibrarianAgent.

        Args:
            image_path: Path to the image file.

        Returns:
            JPEG-encoded bytes, or None on failure.
        """
        try:
            from PIL import Image

            with Image.open(image_path) as img:
                # Convert to RGB (handles RGBA PNGs, grayscale, etc.)
                processed: Image.Image = (
                    img.convert("RGB") if img.mode in ("RGBA", "P", "LA") else img
                )

                # Resize if too large to keep API calls efficient
                max_dim = 1024
                if max(processed.size) > max_dim:
                    processed.thumbnail(
                        (max_dim, max_dim),
                        Image.Resampling.LANCZOS,
                    )

                # Encode as JPEG
                import io

                buf = io.BytesIO()
                processed.save(buf, format="JPEG", quality=85)
                return buf.getvalue()

        except Exception as exc:
            logger.warning("Could not load image %s: %s", image_path, exc)
            return None
