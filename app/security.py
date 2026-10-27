"""Authenticated workspace membership, request-size and error boundaries."""

from dataclasses import dataclass, field
import json
import hmac
import ipaddress
import logging
import os
import re
from tempfile import SpooledTemporaryFile
from urllib.parse import urlsplit
from uuid import uuid4

from starlette.datastructures import Headers
from starlette.responses import JSONResponse


MAX_JSON_BYTES = 128 * 1024
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_FILES = 10
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AccessSettings:
    mode: str = "disabled"
    public_origin: str = ""
    workspace_id: str = "local"
    proxy_secret: str = ""
    proxy_peers: tuple[str, ...] = ("127.0.0.1", "::1")
    allowed_subjects: tuple[str, ...] = ()
    workspace_memberships: dict[str, tuple[str, ...]] = field(default_factory=dict)


def get_access_settings() -> AccessSettings:
    try:
        memberships = json.loads(os.getenv("DOCQA_WORKSPACE_MEMBERSHIPS", "{}"))
        if not isinstance(memberships, dict):
            raise ValueError("Expected an object")
    except (ValueError, TypeError):
        raise BoundaryError(503, "access_not_configured", "Workspace membership is not configured.") from None
    return AccessSettings(
        workspace_memberships=memberships,
        mode=os.getenv("DOCQA_AUTH_MODE", "disabled"),
        public_origin=os.getenv("DOCQA_PUBLIC_ORIGIN", ""),
        workspace_id=os.getenv("DOCQA_WORKSPACE_ID", "local"),
        proxy_secret=os.getenv("DOCQA_PROXY_SECRET", ""),
        proxy_peers=tuple(item.strip() for item in os.getenv("DOCQA_PROXY_PEERS", "127.0.0.1,::1").split(",") if item.strip()),
        allowed_subjects=tuple(item.strip() for item in os.getenv("DOCQA_ALLOWED_SUBJECTS", "").split(",") if item.strip()),
    )


class BoundaryError(Exception):
    def __init__(self, status: int, code: str, detail: str):
        self.status, self.code, self.detail = status, code, detail


def error_response(status: int, code: str, detail: str, request_id: str) -> JSONResponse:
    return JSONResponse({"detail": detail, "code": code, "request_id": request_id}, status_code=status)


def authorize(scope: dict, headers: Headers, settings: AccessSettings) -> dict:
    host = headers.get("host", "").lower()
    peer = (scope.get("client") or ("", 0))[0]
    if headers.get("x-principal-id") or (settings.mode == "local" and headers.get("x-workspace-id")):
        raise BoundaryError(403, "identity_rejected", "Identity must be established by the server.")
    if settings.mode == "local":
        try:
            is_local = ipaddress.ip_address(peer).is_loopback
        except ValueError:
            is_local = False
        if not is_local or not re.fullmatch(r"(?:localhost|127\.0\.0\.1|\[::1\])(?::[0-9]{1,5})?", host):
            raise BoundaryError(403, "local_only", "Local mode accepts loopback requests only.")
        if headers.get("x-auth-request-user") or headers.get("x-docqa-proxy-secret"):
            raise BoundaryError(403, "identity_rejected", "Proxy identity is not accepted in local mode.")
        origin = f"{scope.get('scheme', 'http')}://{host}"
        principal = "local-developer"
        permitted = [settings.workspace_id]
        workspace = settings.workspace_id
    elif settings.mode == "auth_proxy":
        parsed = urlsplit(settings.public_origin)
        if (len(settings.proxy_secret) < 32 or not (settings.allowed_subjects or settings.workspace_memberships)
                or parsed.scheme != "https" or not parsed.netloc or parsed.path
                or parsed.query or parsed.fragment or parsed.username or not settings.workspace_id):
            raise BoundaryError(503, "access_not_configured", "Shared access is not configured.")
        if peer not in settings.proxy_peers or host != parsed.netloc.lower():
            raise BoundaryError(403, "untrusted_proxy", "Request did not arrive through the configured proxy.")
        supplied_secret = headers.get("x-docqa-proxy-secret", "")
        if not hmac.compare_digest(supplied_secret.encode(), settings.proxy_secret.encode()):
            raise BoundaryError(401, "authentication_required", "Authentication is required.")
        principal = headers.get("x-auth-request-user", "")
        memberships = settings.workspace_memberships or {settings.workspace_id: settings.allowed_subjects}
        if (not isinstance(memberships, dict) or any(
                not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", key)
                or not isinstance(members, (list, tuple)) or not members
                or any(not isinstance(member, str) or not member for member in members)
                for key, members in memberships.items())):
            raise BoundaryError(503, "access_not_configured", "Workspace membership is not configured.")
        permitted = sorted(key for key, members in memberships.items() if principal in members)
        workspace = headers.get("x-workspace-id") or (settings.workspace_id if settings.workspace_id in permitted else next(iter(permitted), ""))
        if not principal or workspace not in permitted:
            raise BoundaryError(403, "workspace_forbidden", "This account cannot access the configured workspace.")
        origin = settings.public_origin
    else:
        raise BoundaryError(503, "access_not_configured", "Set DOCQA_AUTH_MODE to configure application access.")
    if headers.get("origin", origin) != origin or headers.get("sec-fetch-site") == "cross-site":
        raise BoundaryError(403, "cross_origin_rejected", "Cross-origin requests are not permitted.")
    return {"principal_id": principal, "workspace_id": workspace, "allowed_workspaces": permitted,
            "can_select_workspace": settings.mode == "auth_proxy"}


class BoundaryMiddleware:
    """Authenticate before reading; spool bounded bodies before multipart/JSON parsing."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = str(uuid4())
        scope.setdefault("state", {})["request_id"] = request_id
        started = False

        async def safe_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                message["headers"] = [(key, value) for key, value in message.get("headers", [])
                                      if key.lower() not in {b"x-request-id", b"cache-control"}]
                message["headers"] += [
                    (b"x-request-id", request_id.encode()), (b"cache-control", b"no-store"),
                    (b"x-content-type-options", b"nosniff"), (b"x-frame-options", b"DENY"),
                    (b"referrer-policy", b"no-referrer"),
                    (b"content-security-policy", b"object-src 'none'; base-uri 'none'; frame-ancestors 'none'"),
                ]
            await send(message)

        try:
            headers = Headers(scope=scope)
            sensitive = {b"host", b"content-length", b"origin", b"x-docqa-proxy-secret", b"x-auth-request-user", b"x-workspace-id", b"x-principal-id"}
            names = [key.lower() for key, _ in scope.get("headers", []) if key.lower() in sensitive]
            if len(names) != len(set(names)):
                raise BoundaryError(400, "ambiguous_headers", "Duplicate request headers are not permitted.")
            context = authorize(scope, headers, get_access_settings())
            scope["state"]["request_context"] = {**context, "request_id": request_id}
            limit = MAX_UPLOAD_BYTES if scope.get("path") == "/api/upload" else MAX_JSON_BYTES
            length = headers.get("content-length")
            if length is not None:
                if not length.isascii() or not length.isdigit():
                    raise BoundaryError(400, "invalid_length", "Invalid Content-Length.")
                if int(length) > limit:
                    raise BoundaryError(413, "request_too_large", "Request exceeds the byte limit.")
            with SpooledTemporaryFile(max_size=1024 * 1024) as body:
                total = 0
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    chunk = message.get("body", b"")
                    total += len(chunk)
                    if total > limit:
                        raise BoundaryError(413, "request_too_large", "Request exceeds the byte limit.")
                    body.write(chunk)
                    if not message.get("more_body", False):
                        break
                if length is not None and int(length) != total:
                    raise BoundaryError(400, "invalid_length", "Content-Length does not match the request body.")
                body.seek(0)
                complete = False

                async def replay():
                    nonlocal complete
                    if complete:
                        return await receive()
                    chunk = body.read(64 * 1024)
                    more = body.tell() < total
                    complete = not more
                    return {"type": "http.request", "body": chunk, "more_body": more}

                await self.app(scope, replay, safe_send)
        except BoundaryError as exc:
            if not started:
                await error_response(exc.status, exc.code, exc.detail, request_id)(scope, receive, safe_send)
        except Exception as exc:
            # Do not log exception text, request bodies, credentials, or source content.
            logger.error("request_failed request_id=%s error_type=%s", request_id, type(exc).__name__)
            if not started:
                await error_response(500, "internal_error", "The request could not be completed.", request_id)(scope, receive, safe_send)
