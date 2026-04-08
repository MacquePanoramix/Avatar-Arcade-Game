"""Stub entrypoint for recording Kinect skeleton sessions."""

from pathlib import Path


def record_session(output_dir: Path, seconds: int = 10) -> None:
    """Placeholder for Kinect capture recording.

    TODO: Implement Kinect stream acquisition and frame serialization.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[TODO] Record Kinect session for {seconds}s -> {output_dir}")
