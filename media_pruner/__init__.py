"""ShutterSort - AI-powered media folder analyzer and pruner.

An agent-based CLI tool that scans local media folders, analyzes them
with a local Vision Language Model (Ollama/llama3.2-vision), detects
duplicates, and provides an interactive cleanup interface.

Architecture:
    - LibrarianAgent: Manages file system, extracts previews/frames, finds duplicates
    - CuratorAgent: Vision analysis via Ollama, returns typed AnalysisResult
    - DecisionAgent: Interactive review, Rich tables, AppleScript trash

Usage:
    shuttersort --path ~/Desktop
    shuttersort --path ~/Photos --model llama3.2-vision
"""

__version__ = "0.1.1"
__author__ = "camiloavilacm"
