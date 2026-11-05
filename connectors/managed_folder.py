"""Bounded read-only adapter over a server-managed source root."""

import hashlib
import mimetypes
from pathlib import Path

from domain.connectors import ConnectorFile, ConnectorScan, validate_source_key


SUPPORTED_SUFFIXES = frozenset({
    ".pdf", ".pptx", ".xlsx", ".xlsm", ".csv", ".tsv", ".txt", ".md", ".json",
    ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp",
})


class ConnectorSourceUnavailable(Exception):
    """Configured source is absent or unsafe."""


class ConnectorLimitError(Exception):
    """Source exceeds bounded connector limits."""


class ManagedFolderAdapter:
    MAX_FILES = 20
    MAX_FILE_BYTES = 2 * 1024 * 1024
    MAX_TOTAL_BYTES = 10 * 1024 * 1024

    def __init__(self, root: Path):
        self.root = Path(root)

    def scan(self, source_key: str) -> ConnectorScan:
        source_key = validate_source_key(source_key)
        if self.root.is_symlink() or not self.root.is_dir():
            raise ConnectorSourceUnavailable()
        source = self.root
        for segment in source_key.split("/"):
            source = source / segment
            if source.is_symlink():
                raise ConnectorSourceUnavailable()
        if not source.is_dir():
            raise ConnectorSourceUnavailable()
        files: list[ConnectorFile] = []
        total = 0
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise ConnectorSourceUnavailable()
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            if len(relative.parts) > 5 or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                raise ConnectorSourceUnavailable()
            if len(files) >= self.MAX_FILES:
                raise ConnectorLimitError()
            file_stat = path.stat()
            if file_stat.st_nlink != 1:
                raise ConnectorSourceUnavailable()
            size = file_stat.st_size
            if size <= 0 or size > self.MAX_FILE_BYTES or total + size > self.MAX_TOTAL_BYTES:
                raise ConnectorLimitError()
            content = path.read_bytes()
            if len(content) != size:
                raise ConnectorSourceUnavailable()
            total += size
            name = relative.as_posix()
            files.append(ConnectorFile(
                name, content, hashlib.sha256(content).hexdigest(),
                mimetypes.guess_type(name)[0] or "application/octet-stream",
            ))
        digest = hashlib.sha256()
        for item in files:
            name = item.relative_path.encode("utf-8")
            digest.update(len(name).to_bytes(4, "big"))
            digest.update(name)
            digest.update(bytes.fromhex(item.sha256))
            digest.update(len(item.content).to_bytes(8, "big"))
        return ConnectorScan(source_key, digest.hexdigest(), tuple(files))
