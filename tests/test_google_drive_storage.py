"""Tests de l'archive Google Drive sans accès réseau réel."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from kata_pipeline.gnn.config import ArchiveConfig
from kata_pipeline.storage.google_drive import (
    FOLDER_MIME_TYPE,
    ArchiveVideoRef,
    GoogleDriveVideoStore,
    load_archive_index,
)


class FakeDriveClient:
    def __init__(self, tree: dict[str, list[dict]], payloads: dict[str, bytes]):
        self.tree = tree
        self.payloads = payloads

    def get_metadata(self, file_id: str) -> dict:
        return {"id": file_id, "name": "data", "mimeType": FOLDER_MIME_TYPE}

    def list_children(self, folder_id: str) -> list[dict]:
        return self.tree.get(folder_id, [])

    def download(self, file_id: str, destination: Path) -> None:
        destination.write_bytes(self.payloads[file_id])


def _folder(file_id: str, name: str) -> dict:
    return {"id": file_id, "name": name, "mimeType": FOLDER_MIME_TYPE}


def _video(file_id: str, name: str, payload: bytes) -> dict:
    return {
        "id": file_id,
        "name": name,
        "mimeType": "video/mp4",
        "size": str(len(payload)),
        "md5Checksum": hashlib.md5(payload).hexdigest(),  # noqa: S324
        "modifiedTime": "2026-09-03T10:00:00Z",
        "capabilities": {"canDownload": True},
    }


def _config(tmp_path: Path) -> ArchiveConfig:
    return ArchiveConfig(
        enabled=True,
        root_folder_id="root",
        remote_clips_path="clips",
        index_file=tmp_path / "archive.csv",
        cache_dir=tmp_path / "cache",
        cache_max_gb=1.0,
    )


def test_recursive_index_preserves_ids_types_and_pending(tmp_path: Path) -> None:
    valid = b"valid-video"
    pending = b"pending-video"
    tree = {
        "root": [_folder("clips-id", "clips"), _folder("viz-id", "viz")],
        "clips-id": [_folder("sa-id", "SA"), _folder("k1-id", "K1")],
        "sa-id": [_folder("pool-id", "pool_1"), _folder("pending-id", "pending")],
        "pool-id": [_video("valid-id", "SA_Test_Female_T1_m01_red_A_Anan.mp4", valid)],
        "pending-id": [_video("pending-video-id", "SA_pending.mp4", pending)],
        "k1-id": [],
    }
    client = FakeDriveClient(
        tree, {"valid-id": valid, "pending-video-id": pending}
    )
    config = _config(tmp_path)
    store = GoogleDriveVideoStore(config, client)

    refs = store.rebuild_index()
    loaded = load_archive_index(config.index_file)

    assert len(refs) == len(loaded) == 2
    indexed = next(ref for ref in loaded if ref.drive_file_id == "valid-id")
    assert indexed.competition_type == "SA"
    assert indexed.relative_path.startswith("clips/SA/pool_1/")
    assert indexed.eligible_for_pose_extraction
    assert not next(ref for ref in loaded if ref.is_pending).eligible_for_pose_extraction


def test_download_is_verified_and_writes_provenance(tmp_path: Path) -> None:
    payload = b"video-from-drive"
    ref = ArchiveVideoRef(
        provider="google_drive",
        root_folder_id="root",
        drive_file_id="video-id",
        name="SA_Test.mp4",
        stem="SA_Test",
        relative_path="clips/SA/pool_1/SA_Test.mp4",
        competition_type="SA",
        mime_type="video/mp4",
        size=len(payload),
        md5_checksum=hashlib.md5(payload).hexdigest(),  # noqa: S324
        modified_time="2026-09-03T10:00:00Z",
    )
    store = GoogleDriveVideoStore(
        _config(tmp_path), FakeDriveClient({}, {"video-id": payload})
    )

    path = store.resolve(ref)
    metadata = json.loads(path.with_suffix(".mp4.meta.json").read_text(encoding="utf-8"))

    assert path.read_bytes() == payload
    assert path.parent.name == "SA"
    assert metadata["drive_file_id"] == "video-id"
    assert metadata["md5"] == ref.md5_checksum
    assert metadata["sha256"] == hashlib.sha256(payload).hexdigest()


def test_lru_prune_removes_oldest_video(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.cache_max_gb = 10 / 1024**3
    store = GoogleDriveVideoStore(config, FakeDriveClient({}, {}))
    old = config.cache_dir / "SA" / "old.mp4"
    recent = config.cache_dir / "SA" / "recent.mp4"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"123456")
    recent.write_bytes(b"abcdef")
    os.utime(old, (1, 1))
    os.utime(recent, (2, 2))

    removed = store.prune()

    assert removed == [old]
    assert not old.exists()
    assert recent.exists()
