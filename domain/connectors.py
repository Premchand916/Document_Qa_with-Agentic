"""Connector contracts. Remote data is untrusted evidence, never authority."""

from dataclasses import dataclass
from enum import Enum
import re


IDENTIFIER = re.compile(r"^[a-f0-9]{32}$")
SOURCE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*){0,4}$")


class ConnectorType(str, Enum):
    MANAGED_FOLDER = "managed_folder"


class ConnectorState(str, Enum):
    ACTIVE = "active"
    REVOKED = "revoked"


class SyncState(str, Enum):
    RUNNING = "running"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True)
class ConnectorFile:
    relative_path: str
    content: bytes
    sha256: str
    media_type: str


@dataclass(frozen=True)
class ConnectorScan:
    source_key: str
    cursor: str
    files: tuple[ConnectorFile, ...]


def validate_source_key(value: str) -> str:
    normalized = value.strip()
    if not SOURCE_KEY.fullmatch(normalized):
        raise ValueError("Invalid connector source key")
    return normalized


def validate_display_name(value: str) -> str:
    normalized = " ".join(value.strip().split())
    if not 1 <= len(normalized) <= 120:
        raise ValueError("Connector display name must contain 1 to 120 characters")
    return normalized
