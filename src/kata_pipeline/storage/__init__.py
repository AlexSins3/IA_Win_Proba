"""Stockages externes et cache local des artefacts vidéo."""

from kata_pipeline.storage.google_drive import (
    ArchiveVideoRef,
    GoogleDriveVideoStore,
    load_archive_index,
)

__all__ = ["ArchiveVideoRef", "GoogleDriveVideoStore", "load_archive_index"]
