"""Base agent class with Ollama interface and automatic retry logic.

This module defines the MediaAgent abstract base class (ABC) that provides:
    - Ollama API connection management
    - Automatic retry with "reflection" on JSON parse failures
    - Shared logging and configuration

The retry loop is a key "agentic" feature: if the LLM returns malformed JSON,
instead of crashing, the agent sends the error back to the model and asks it
to self-correct. This mimics how a human would say "that didn't make sense,
try again."
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from .models import AnalysisResult
from .utils import parse_json_with_retry

# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------
# We use Python's built-in logging module instead of print() because:
# - Logs can be directed to files, stdout, or both
# - Log levels (DEBUG, INFO, WARNING, ERROR) provide filtering control
# - Each log entry includes timestamp, level, and source module
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


class MediaAgent(ABC):
    """Abstract base class for all agents in the ShutterSort system.

    This class provides the shared infrastructure that all agents need:
    - Connection to Ollama
    - Retry logic for LLM calls
    - Logging

    Subclasses must implement their specific behavior via abstract methods.
    This is the Template Method pattern: the base class defines the skeleton
    of an algorithm (the retry loop), and subclasses fill in the details.

    Attributes:
        model: The Ollama model name (e.g., 'llama3.2-vision').
        max_retries: Maximum number of retry attempts on JSON parse failure.
        ollama_client: The Ollama client instance (set in __init__).
    """

    def __init__(
        self,
        model: str = "llama3.2-vision",
        max_retries: int = 3,
        ollama_client: Any = None,
    ) -> None:
        """Initialize the MediaAgent.

        Args:
            model: The Ollama model to use for vision analysis.
            max_retries: How many times to retry on JSON parse failure.
            ollama_client: Optional pre-configured Ollama client (for testing).
                          If None, a new ollama.Client() is created.
        """
        self.model = model
        self.max_retries = max_retries

        # Lazy import of ollama to avoid import errors when the package
        # isn't installed yet (e.g., during `pip install` phase).
        if ollama_client is not None:
            self.ollama_client = ollama_client
        else:
            import ollama

            self.ollama_client = ollama.Client()

    # -----------------------------------------------------------------------
    # Retry loop with reflection
    # -----------------------------------------------------------------------
    def call_ollama_with_retry(
        self,
        prompt: str,
        images: list[bytes] | None = None,
    ) -> AnalysisResult:
        """Call Ollama with automatic retry on JSON parse failure.

        This is the core "agentic" behavior. The flow is:
        1. Send prompt + images to Ollama
        2. Try to parse the response as JSON
        3. If parsing fails, send the error back to Ollama and retry
        4. Repeat up to max_retries times
        5. If all retries fail, raise the last exception

        The "reflection" happens in step 3: we tell the model exactly what
        went wrong ("Invalid JSON: ...") so it can self-correct. This is
        much more effective than a blind retry.

        Args:
            prompt: The text prompt to send to the model.
            images: Optional list of image bytes (JPEG-encoded).

        Returns:
            A typed AnalysisResult with the model's analysis.

        Raises:
            ValueError: If all retries fail to produce valid JSON.
            Exception: If Ollama itself fails (network error, etc.).
        """
        last_error: Exception | None = None
        current_prompt = prompt

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    "Calling Ollama (attempt %d/%d, model=%s)",
                    attempt,
                    self.max_retries,
                    self.model,
                )

                # Build the message for Ollama
                # The ollama Python client expects images inside the message dict
                message: dict[str, Any] = {
                    "role": "user",
                    "content": current_prompt,
                }

                if images:
                    message["images"] = images

                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": [message],
                }

                # Call the Ollama API
                response = self.ollama_client.chat(**kwargs)
                response_text: str = response["message"]["content"]

                # Try to parse the response as JSON
                parsed = parse_json_with_retry(response_text)

                # Convert the parsed dict to an AnalysisResult
                # Using .get() with defaults provides safety against missing fields
                result = AnalysisResult(
                    scene_type=parsed.get("scene_type", "other"),
                    score=int(parsed.get("score", 1)),
                    summary=parsed.get("summary", ""),
                    people_count=int(parsed.get("people_count", 0)),
                    people_description=parsed.get("people_description", ""),
                    emotions_detected=parsed.get("emotions_detected", ""),
                    raw_json=response_text,
                )

                logger.info(
                    "Ollama response parsed successfully: scene=%s, score=%d",
                    result.scene_type,
                    result.score,
                )
                return result

            except (ValueError, KeyError) as exc:
                # JSON parse error or missing field — retry with reflection
                last_error = exc
                logger.warning(
                    "Attempt %d failed: %s. Retrying with error feedback...",
                    attempt,
                    exc,
                )

                # "Reflect" the error back to the model
                # This tells the model what went wrong and asks it to fix it
                current_prompt = (
                    f"{prompt}\n\n"
                    f"ERROR: Your previous response could not be parsed. "
                    f"Details: {exc}\n\n"
                    f"Please respond with ONLY a valid JSON object matching "
                    f"the required schema. Do NOT include any text before or "
                    f"after the JSON. Do NOT use markdown code blocks."
                )

        # All retries exhausted
        raise ValueError(
            f"Failed to get valid JSON from Ollama after {self.max_retries} "
            f"attempts. Last error: {last_error}"
        ) from last_error

    # -----------------------------------------------------------------------
    # Abstract method: each agent defines its own execution logic
    # -----------------------------------------------------------------------
    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the agent's primary task.

        Each concrete agent implements this method with its specific logic.
        This is the entry point for the agent's work.

        Args:
            *args: Positional arguments specific to the agent.
            **kwargs: Keyword arguments specific to the agent.

        Returns:
            The result of the agent's execution (type varies by agent).
        """
        ...
