"""Connector orchestration with authorization, freshness, and safe failures."""

from collections.abc import Mapping
import time

from connectors.managed_folder import (ConnectorLimitError, ConnectorSourceUnavailable,
                                       ManagedFolderAdapter)
from storage.connectors import (ConnectorCapacityError, ConnectorConflict,
                                ConnectorRepository, ConnectorRevoked)


class ConnectorAccessDenied(Exception):
    """Principal is not authorized for the server-managed source key."""


class ConnectorConfigurationError(Exception):
    """Connector runtime is not configured safely."""


class ConnectorSyncError(Exception):
    """A sync failed with a stable, non-sensitive error code."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class ConnectorService:
    def __init__(self, repository: ConnectorRepository, adapter: ManagedFolderAdapter,
                 *, source_acl: Mapping[str, frozenset[str]] | None = None,
                 allow_unlisted_principals: frozenset[str] = frozenset(),
                 configured: bool = True, freshness_seconds: int = 24 * 60 * 60):
        if not 60 <= freshness_seconds <= 30 * 24 * 60 * 60:
            raise ValueError("Connector freshness must be between 60 seconds and 30 days")
        self.repository = repository
        self.adapter = adapter
        self.source_acl = dict(source_acl or {})
        self.allow_unlisted_principals = allow_unlisted_principals
        self.configured = configured
        self.freshness_seconds = freshness_seconds

    def _authorize(self, principal: str, source_key: str) -> None:
        if not self.configured:
            raise ConnectorConfigurationError()
        permitted = self.source_acl.get(source_key, frozenset())
        if principal not in self.allow_unlisted_principals and principal not in permitted:
            raise ConnectorAccessDenied()

    def create_source(self, workspace: str, principal: str, source_key: str,
                      display_name: str, idempotency_key: str) -> dict:
        self._authorize(principal, source_key)
        return self.repository.create_source(
            workspace, principal, source_key, display_name, idempotency_key,
        )

    def list_sources(self, workspace: str, principal: str,
                     *, now: int | None = None) -> list[dict]:
        current_time = int(time.time() if now is None else now)
        result = []
        for source in self.repository.list_sources(workspace, principal):
            if source["state"] != "active" or source["grant_state"] != "active":
                freshness = "revoked"
            elif source["last_synced_at"] is None:
                freshness = "never_synced"
            elif current_time - source["last_synced_at"] <= self.freshness_seconds:
                freshness = "fresh"
            else:
                freshness = "stale"
            publication = ("revoked" if freshness == "revoked" else
                           "needs_sync" if source["last_snapshot_id"] is None else
                           "ready" if source["published_snapshot_id"] == source["last_snapshot_id"] else
                           "needs_publish")
            result.append({**source, "freshness": freshness,
                           "publication_state": publication})
        return result

    def authorized_published_sources(self, workspace: str, principal: str) -> list[dict]:
        return [source for source in self.list_sources(workspace, principal)
                if source["publication_state"] == "ready" and source["freshness"] == "fresh"
                and self.can_access(principal, source["source_key"])]

    def can_access(self, principal: str, source_key: str) -> bool:
        return self.configured and (principal in self.allow_unlisted_principals
                                    or principal in self.source_acl.get(source_key, frozenset()))

    def sync(self, workspace: str, principal: str, source_id: str,
             idempotency_key: str) -> dict:
        source = self.repository.get_source(workspace, principal, source_id)
        self._authorize(principal, source["source_key"])
        run, created = self.repository.begin_sync(
            workspace, principal, source_id, idempotency_key,
        )
        if not created:
            return run
        try:
            scan = self.adapter.scan(source["source_key"])
            return self.repository.complete_sync(workspace, principal, run["run_id"], scan)
        except ConnectorLimitError as exc:
            self._record_failure(workspace, principal, run["run_id"], "source_limit")
            raise ConnectorSyncError("source_limit") from exc
        except ConnectorSourceUnavailable as exc:
            self._record_failure(workspace, principal, run["run_id"], "source_unavailable")
            raise ConnectorSyncError("source_unavailable") from exc
        except ConnectorRevoked as exc:
            self._record_failure(workspace, principal, run["run_id"], "source_revoked")
            raise
        except ConnectorCapacityError as exc:
            self._record_failure(workspace, principal, run["run_id"], "snapshot_capacity")
            raise ConnectorSyncError("snapshot_capacity") from exc
        except Exception as exc:
            self._record_failure(workspace, principal, run["run_id"], "sync_failed")
            raise ConnectorSyncError("sync_failed") from exc

    def _record_failure(self, workspace: str, principal: str, run_id: str,
                        error_code: str) -> None:
        try:
            self.repository.fail_sync(workspace, principal, run_id, error_code)
        except ConnectorConflict:
            pass

    def list_snapshots(self, workspace: str, principal: str,
                       source_id: str, *, limit: int = 20) -> list[dict]:
        source = self.repository.get_source(workspace, principal, source_id)
        self._authorize(principal, source["source_key"])
        return self.repository.list_snapshots(workspace, principal, source_id, limit=limit)

    def revoke_grant(self, workspace: str, principal: str, grant_id: str) -> dict:
        return self.repository.revoke_grant(workspace, principal, grant_id)

    def delete_source(self, workspace: str, principal: str, source_id: str) -> dict:
        return self.repository.delete_source(workspace, principal, source_id)
