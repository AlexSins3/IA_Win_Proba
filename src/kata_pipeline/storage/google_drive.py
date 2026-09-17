"""Archive vidéo Google Drive en lecture seule avec cache local borné."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from kata_pipeline.competition_formats import (
    COMPETITION_FORMATS,
    CompetitionType,
    normalize_competition_type,
    typed_competition_dir,
)
from kata_pipeline.gnn.config import ArchiveConfig

logger = logging.getLogger(__name__)

FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi"}
DRIVE_READONLY_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
INDEX_COLUMNS = [
    "provider",
    "root_folder_id",
    "drive_file_id",
    "name",
    "stem",
    "relative_path",
    "competition_type",
    "mime_type",
    "size",
    "md5_checksum",
    "modified_time",
    "can_download",
]


@dataclass(frozen=True)
class ArchiveVideoRef:
    """Référence stable vers une vidéo binaire archivée sur Drive."""

    provider: str
    root_folder_id: str
    drive_file_id: str
    name: str
    stem: str
    relative_path: str
    competition_type: CompetitionType | None
    mime_type: str
    size: int
    md5_checksum: str
    modified_time: str
    can_download: bool = True

    @property
    def is_pending(self) -> bool:
        return "pending" in {part.lower() for part in PurePosixPath(self.relative_path).parts}

    @property
    def is_derived(self) -> bool:
        return self.stem.endswith(("_motion", "_saliency"))

    @property
    def eligible_for_pose_extraction(self) -> bool:
        return not self.is_pending and not self.is_derived


class DriveClient(Protocol):
    """Surface minimale de l'API, injectable dans les tests."""

    def get_metadata(self, file_id: str) -> dict[str, Any]: ...

    def list_children(self, folder_id: str) -> list[dict[str, Any]]: ...

    def download(self, file_id: str, destination: Path) -> None: ...


class GoogleDriveClient:
    """Client Google Drive v3 authentifié par OAuth utilisateur."""

    def __init__(self, service: Any):
        self.service = service

    @classmethod
    def authenticate(cls, config: ArchiveConfig, *, interactive: bool = True):
        """Charge le token local ou ouvre le consentement OAuth au premier usage."""

        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
        except ImportError as exc:
            raise RuntimeError(
                "Dépendances Google Drive absentes. Installer avec : "
                "pip install -e \".[drive]\""
            ) from exc

        credentials = None
        if config.token_file.exists():
            credentials = Credentials.from_authorized_user_file(
                str(config.token_file), [DRIVE_READONLY_SCOPE]
            )
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        if not credentials or not credentials.valid:
            if not interactive:
                raise RuntimeError(
                    f"Token OAuth absent ou invalide : {config.token_file}. "
                    "Lancer d'abord `kata-pipeline drive auth`."
                )
            if not config.credentials_file.exists():
                raise FileNotFoundError(
                    f"Credentials OAuth introuvables : {config.credentials_file}"
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(config.credentials_file), [DRIVE_READONLY_SCOPE]
            )
            credentials = flow.run_local_server(port=0)
        config.token_file.parent.mkdir(parents=True, exist_ok=True)
        config.token_file.write_text(credentials.to_json(), encoding="utf-8")
        service = build("drive", "v3", credentials=credentials, cache_discovery=False)
        return cls(service)

    def get_metadata(self, file_id: str) -> dict[str, Any]:
        return (
            self.service.files()
            .get(
                fileId=file_id,
                fields="id,name,mimeType,size,md5Checksum,modifiedTime,capabilities(canDownload)",
                supportsAllDrives=True,
            )
            .execute()
        )

    def list_children(self, folder_id: str) -> list[dict[str, Any]]:
        files: list[dict[str, Any]] = []
        page_token = None
        while True:
            response = (
                self.service.files()
                .list(
                    q=f"'{folder_id}' in parents and trashed = false",
                    spaces="drive",
                    pageSize=1000,
                    pageToken=page_token,
                    fields=(
                        "nextPageToken,files(id,name,mimeType,size,md5Checksum,"
                        "modifiedTime,capabilities(canDownload))"
                    ),
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            files.extend(response.get("files", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                return files

    def download(self, file_id: str, destination: Path) -> None:
        try:
            from googleapiclient.http import MediaIoBaseDownload
        except ImportError as exc:
            raise RuntimeError("Dépendances Google Drive absentes.") from exc

        request = self.service.files().get_media(fileId=file_id)
        with destination.open("wb") as stream:
            downloader = MediaIoBaseDownload(stream, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    logger.info("Téléchargement Drive : %.0f %%", status.progress() * 100)


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() not in {"", "0", "false", "no", "none"}


def _from_csv_row(row: dict[str, str]) -> ArchiveVideoRef:
    competition_type = normalize_competition_type(row.get("competition_type"))
    return ArchiveVideoRef(
        provider=row.get("provider", "google_drive"),
        root_folder_id=row.get("root_folder_id", ""),
        drive_file_id=row["drive_file_id"],
        name=row["name"],
        stem=row["stem"],
        relative_path=row["relative_path"],
        competition_type=competition_type,
        mime_type=row.get("mime_type", "video/mp4"),
        size=int(row.get("size") or 0),
        md5_checksum=row.get("md5_checksum", ""),
        modified_time=row.get("modified_time", ""),
        can_download=_as_bool(row.get("can_download", "true")),
    )


def load_archive_index(path: Path) -> list[ArchiveVideoRef]:
    """Charge le manifeste Drive sans effectuer d'appel réseau."""

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [_from_csv_row(row) for row in csv.DictReader(stream)]


def save_archive_index(path: Path, refs: list[ArchiveVideoRef]) -> Path:
    """Écrit l'index local déterministe des références Drive."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        for ref in sorted(refs, key=lambda item: item.relative_path.lower()):
            row = asdict(ref)
            row["competition_type"] = ref.competition_type or ""
            writer.writerow(row)
    return path


def index_by_stem(
    refs: list[ArchiveVideoRef],
    *,
    competition_type: CompetitionType | None = None,
    extraction_only: bool = True,
) -> dict[str, ArchiveVideoRef]:
    """Indexe les vidéos par stem et refuse les références ambiguës."""

    requested_type = normalize_competition_type(competition_type)
    result: dict[str, ArchiveVideoRef] = {}
    for ref in refs:
        if extraction_only and not ref.eligible_for_pose_extraction:
            continue
        if requested_type is not None and ref.competition_type != requested_type:
            continue
        previous = result.get(ref.stem)
        if previous is not None and previous.drive_file_id != ref.drive_file_id:
            raise ValueError(
                f"Stem Drive ambigu {ref.stem!r} : {previous.relative_path} / "
                f"{ref.relative_path}"
            )
        result[ref.stem] = ref
    return result


def _competition_type_from_remote_path(path: PurePosixPath) -> CompetitionType | None:
    for part in path.parts:
        candidate = part.upper()
        if candidate in COMPETITION_FORMATS:
            return normalize_competition_type(candidate)
    return normalize_competition_type(None, path.stem)


def _hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GoogleDriveVideoStore:
    """Indexe l'archive Drive et restaure une vidéo dans un cache LRU."""

    def __init__(self, config: ArchiveConfig, client: DriveClient | None = None):
        self.config = config
        self._client = client

    @property
    def max_cache_bytes(self) -> int:
        return int(self.config.cache_max_gb * 1024**3)

    def client(self, *, interactive: bool = False) -> DriveClient:
        if self._client is None:
            self._client = GoogleDriveClient.authenticate(
                self.config, interactive=interactive
            )
        return self._client

    def authenticate(self) -> dict[str, Any]:
        """Déclenche OAuth puis retourne les métadonnées du dossier racine."""

        if not self.config.root_folder_id:
            raise ValueError("archive.root_folder_id n'est pas configuré")
        return self.client(interactive=True).get_metadata(self.config.root_folder_id)

    def _resolve_subfolder(self, root_id: str, relative_path: str) -> tuple[str, str]:
        folder_id = root_id
        clean_parts = [part for part in PurePosixPath(relative_path).parts if part not in {".", ""}]
        traversed: list[str] = []
        for part in clean_parts:
            candidates = [
                item
                for item in self.client().list_children(folder_id)
                if item.get("mimeType") == FOLDER_MIME_TYPE and item.get("name") == part
            ]
            if len(candidates) != 1:
                raise FileNotFoundError(
                    f"Dossier Drive introuvable ou ambigu sous {'/'.join(traversed) or '/'} : "
                    f"{part!r}"
                )
            folder_id = str(candidates[0]["id"])
            traversed.append(part)
        return folder_id, "/".join(traversed)

    def rebuild_index(self) -> list[ArchiveVideoRef]:
        """Parcourt récursivement ``data/clips`` sur Drive et sauvegarde l'index."""

        root_id = self.config.root_folder_id
        if not root_id:
            raise ValueError("archive.root_folder_id n'est pas configuré")
        clips_id, prefix = self._resolve_subfolder(root_id, self.config.remote_clips_path)
        refs: list[ArchiveVideoRef] = []

        def walk(folder_id: str, relative: PurePosixPath) -> None:
            for item in self.client().list_children(folder_id):
                child_path = relative / str(item["name"])
                if item.get("mimeType") == FOLDER_MIME_TYPE:
                    walk(str(item["id"]), child_path)
                    continue
                if child_path.suffix.lower() not in VIDEO_EXTENSIONS:
                    continue
                capabilities = item.get("capabilities") or {}
                refs.append(
                    ArchiveVideoRef(
                        provider="google_drive",
                        root_folder_id=root_id,
                        drive_file_id=str(item["id"]),
                        name=str(item["name"]),
                        stem=child_path.stem,
                        relative_path=child_path.as_posix(),
                        competition_type=_competition_type_from_remote_path(child_path),
                        mime_type=str(item.get("mimeType") or "application/octet-stream"),
                        size=int(item.get("size") or 0),
                        md5_checksum=str(item.get("md5Checksum") or ""),
                        modified_time=str(item.get("modifiedTime") or ""),
                        can_download=bool(capabilities.get("canDownload", True)),
                    )
                )

        walk(clips_id, PurePosixPath(prefix))
        index_by_stem(refs)
        save_archive_index(self.config.index_file, refs)
        logger.info("Index Drive sauvegardé : %d vidéos -> %s", len(refs), self.config.index_file)
        return refs

    def references(self) -> list[ArchiveVideoRef]:
        return load_archive_index(self.config.index_file)

    def find(
        self, stem: str, competition_type: CompetitionType | None = None
    ) -> ArchiveVideoRef | None:
        return index_by_stem(
            self.references(), competition_type=competition_type
        ).get(stem)

    def cache_path(self, ref: ArchiveVideoRef) -> Path:
        parent = typed_competition_dir(self.config.cache_dir, ref.competition_type)
        return parent / ref.name

    def _cache_files(self) -> list[Path]:
        if not self.config.cache_dir.exists():
            return []
        return [
            path
            for path in self.config.cache_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in VIDEO_EXTENSIONS
            and not path.name.endswith(".part")
        ]

    def cache_size_bytes(self) -> int:
        return sum(path.stat().st_size for path in self._cache_files())

    def prune(self, *, incoming_bytes: int = 0, protected: Path | None = None) -> list[Path]:
        """Évince les vidéos les moins récemment utilisées jusqu'à la limite."""

        if incoming_bytes > self.max_cache_bytes:
            raise ValueError(
                f"Vidéo de {incoming_bytes / 1024**3:.2f} Go supérieure au cache "
                f"de {self.config.cache_max_gb:.2f} Go"
            )
        files = sorted(self._cache_files(), key=lambda path: path.stat().st_mtime)
        current = sum(path.stat().st_size for path in files)
        removed: list[Path] = []
        for path in files:
            if current + incoming_bytes <= self.max_cache_bytes:
                break
            if protected is not None and path == protected:
                continue
            size = path.stat().st_size
            path.unlink()
            metadata_path = path.with_suffix(path.suffix + ".meta.json")
            if metadata_path.exists():
                metadata_path.unlink()
            current -= size
            removed.append(path)
        return removed

    def resolve(self, ref: ArchiveVideoRef) -> Path:
        """Retourne le cache existant ou télécharge et vérifie la vidéo."""

        if not ref.can_download:
            raise PermissionError(f"Téléchargement interdit par Drive : {ref.relative_path}")
        destination = self.cache_path(ref)
        if destination.exists() and (not ref.size or destination.stat().st_size == ref.size):
            os.utime(destination, None)
            return destination

        destination.parent.mkdir(parents=True, exist_ok=True)
        self.prune(incoming_bytes=ref.size, protected=destination)
        partial = destination.with_suffix(destination.suffix + ".part")
        if partial.exists():
            partial.unlink()
        try:
            self.client().download(ref.drive_file_id, partial)
            if ref.size and partial.stat().st_size != ref.size:
                raise IOError(
                    f"Taille invalide pour {ref.name}: {partial.stat().st_size} != {ref.size}"
                )
            md5 = _hash_file(partial, "md5")
            if self.config.verify_md5 and ref.md5_checksum and md5 != ref.md5_checksum:
                raise IOError(f"Checksum MD5 invalide pour {ref.name}")
            sha256 = _hash_file(partial, "sha256")
            partial.replace(destination)
            metadata = {
                "provider": ref.provider,
                "drive_file_id": ref.drive_file_id,
                "remote_path": ref.relative_path,
                "size": destination.stat().st_size,
                "md5": md5,
                "sha256": sha256,
                "downloaded_at": datetime.now(UTC).isoformat(),
            }
            destination.with_suffix(destination.suffix + ".meta.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
            os.utime(destination, None)
            return destination
        except Exception:
            if partial.exists():
                partial.unlink()
            raise

    def resolve_stem(
        self, stem: str, competition_type: CompetitionType | None = None
    ) -> Path:
        ref = self.find(stem, competition_type)
        if ref is None:
            raise FileNotFoundError(f"Vidéo absente de l'index Drive : {stem}")
        return self.resolve(ref)
