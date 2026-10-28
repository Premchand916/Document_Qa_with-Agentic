"""Desired Step 1 behavior; tests must pass without models or external identity services."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from tests.support import OfflineCase


class InputBoundaryTests(OfflineCase):
    def setUp(self):
        super().setUp()
        self.invocations = []
        self.seed_workspace()
        self.server._graph = SimpleNamespace(invoke=lambda state: self.invocations.append(state) or {"response": "ok"})

    def test_blank_query_is_rejected_before_model_work(self):
        for query in ("", "  \n\t"):
            self.assertEqual(self.client().post("/api/chat", json={"query": query}).status_code, 422)
        self.assertEqual(self.invocations, [])

    def test_null_or_forged_history_roles_are_rejected(self):
        for role in (None, "system", "developer", "tool"):
            response = self.client().post("/api/chat", json={"query": "alpha", "chat_history": [{"role": role, "content": "text"}]})
            self.assertEqual(response.status_code, 422)
        self.assertEqual(self.invocations, [])

    def test_history_and_query_limits(self):
        for payload in ({"query": "x" * 8001}, {"query": "alpha", "chat_history": [{"role": "user", "content": "x"}] * 41},
                        {"query": "alpha", "chat_history": [{"role": "user", "content": "x" * 8001}]}):
            self.assertEqual(self.client().post("/api/chat", json=payload).status_code, 422)

    def test_unknown_controls_and_client_identity_fields_are_rejected(self):
        for extras in ({"audience": "invented"}, {"answer_mode": "invented"}, {"response_depth": "invented"},
                       {"workspace_id": "victim"}, {"principal_id": "admin"}, {"use_web_search": "false"}):
            self.assertEqual(self.client().post("/api/chat", json={"query": "alpha", **extras}).status_code, 422)

    def test_client_history_is_validated_but_not_trusted_as_conversation_memory(self):
        response = self.client().post("/api/chat", json={"query": " alpha ", "chat_history": [{"role": "user", "content": "previous"}]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.invocations[0]["query"], "alpha")
        self.assertEqual(self.invocations[0]["chat_history"], [])

    def test_validation_errors_do_not_echo_sensitive_input(self):
        secret = "SYNTHETIC_PRIVATE_INPUT"
        response = self.client().post("/api/chat", json={"query": secret, "audience": secret})
        self.assertEqual(response.status_code, 422)
        self.assertNotIn(secret, response.text)
        self.assertEqual(response.json()["request_id"], response.headers["x-request-id"])

    def test_unexpected_exception_is_redacted_with_request_id(self):
        self.server._graph = SimpleNamespace(invoke=lambda state: (_ for _ in ()).throw(RuntimeError("SYNTHETIC_SECRET_PATH")))
        response = self.client().post("/api/chat", json={"query": "alpha"})
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("SYNTHETIC_SECRET_PATH", response.text)
        self.assertEqual(response.json()["code"], "internal_error")
        self.assertEqual(response.json()["request_id"], response.headers["x-request-id"])

    def test_provider_error_does_not_leak_configuration(self):
        import agents.qa_agent as qa
        from langchain_core.documents import Document
        with patch.object(qa, "invoke_orchestrator", side_effect=qa.LLMConfigurationError("SYNTHETIC_PROVIDER_SECRET")):
            self.server._graph = SimpleNamespace(invoke=lambda state: qa.qa_agent({**state, "documents": [Document(page_content="alpha")]}))
            response = self.client().post("/api/chat", json={"query": "alpha"})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("SYNTHETIC_PROVIDER_SECRET", response.text)


class UploadBoundaryTests(OfflineCase):
    def test_file_count_limit_precedes_document_parsing(self):
        with patch.object(self.server, "parse_uploaded_file") as loader:
            response = self.client().post("/api/upload", files=[("files", (f"{i}.txt", b"alpha")) for i in range(11)])
        self.assertEqual(response.status_code, 413)
        loader.assert_not_called()

    def test_file_size_limit_precedes_document_parsing(self):
        with patch("app.server.MAX_FILE_BYTES", 4), patch.object(self.server, "parse_uploaded_file") as loader:
            response = self.client().post("/api/upload", files={"files": ("large.txt", b"12345")})
        self.assertEqual(response.status_code, 413)
        loader.assert_not_called()

    def test_all_file_sizes_checked_before_any_file_is_parsed(self):
        with patch("app.server.MAX_FILE_BYTES", 4), patch.object(self.server, "parse_uploaded_file") as loader:
            response = self.client().post("/api/upload", files=[("files", ("ok.txt", b"123")), ("files", ("large.txt", b"12345"))])
        self.assertEqual(response.status_code, 413)
        loader.assert_not_called()

    def test_empty_or_path_filename_is_rejected(self):
        for name in ("../private.txt", "folder\\private.txt"):
            response = self.client().post("/api/upload", files={"files": (name, b"alpha")})
            self.assertEqual(response.status_code, 422)

    def test_parser_exception_is_redacted(self):
        with patch.object(self.server, "parse_uploaded_file", side_effect=ValueError("SYNTHETIC_PARSER_SECRET")):
            response = self.client().post("/api/upload", files={"files": ("bad.pdf", b"not pdf")})
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("SYNTHETIC_PARSER_SECRET", response.text)

    def test_ocr_and_parser_limits_have_actionable_safe_errors(self):
        from ingestion.file_loader import IngestionLimitError, OCRConfigurationError, OCRRequiredError
        for failure, status, code in ((OCRRequiredError("private path"), 422, "ocr_required"),
                                      (IngestionLimitError("private limit detail"), 413, "request_too_large"),
                                      (OCRConfigurationError("private language"), 503, "ocr_not_configured")):
            with patch.object(self.server, "parse_uploaded_file", side_effect=failure):
                response = self.client().post("/api/upload", files={"files": ("scan.pdf", b"synthetic")})
            self.assertEqual(response.status_code, status)
            self.assertEqual(response.json()["code"], code)
            self.assertNotIn("private", response.text)

    def test_json_body_limit_precedes_route(self):
        response = self.client().post("/api/chat", content=b"x" * (128 * 1024 + 1), headers={"Content-Type": "application/json"})
        self.assertEqual(response.status_code, 413)

    def test_chunked_body_cannot_bypass_limit(self):
        from app.security import BoundaryMiddleware
        called = []
        async def endpoint(scope, receive, send):
            called.append(True)
        scope = {"type": "http", "method": "POST", "path": "/api/chat", "scheme": "http", "client": ("127.0.0.1", 1),
                 "server": ("127.0.0.1", 80), "headers": [(b"host", b"127.0.0.1")]}
        messages = iter([{"type": "http.request", "body": b"x" * (64 * 1024), "more_body": True}] * 3)
        output = []
        async def receive():
            return next(messages)
        async def send(message):
            output.append(message)
        asyncio.run(BoundaryMiddleware(endpoint)(scope, receive, send))
        self.assertEqual(output[0]["status"], 413)
        self.assertEqual(called, [])


class AccessBoundaryTests(OfflineCase):
    def access(self, **changes):
        from app.security import AccessSettings
        config = replace(AccessSettings(mode="local"), **changes)
        return patch("app.security.get_access_settings", return_value=config)

    def test_disabled_mode_fails_closed(self):
        with self.access(mode="disabled"):
            self.assertEqual(self.client().get("/api/state").status_code, 503)

    def test_local_mode_rejects_nonlocal_peer_and_rebinding_host(self):
        from fastapi.testclient import TestClient
        with self.access(), TestClient(self.server.app, client=("203.0.113.1", 123), base_url="http://127.0.0.1") as client:
            self.assertEqual(client.get("/api/state").status_code, 403)
        with self.access():
            self.assertEqual(self.client().get("/api/state", headers={"host": "evil.example"}).status_code, 403)

    def test_cross_site_requests_and_spoofed_identity_are_rejected(self):
        for headers in ({"origin": "https://evil.example"}, {"sec-fetch-site": "cross-site"},
                        {"x-workspace-id": "victim"}, {"x-auth-request-user": "admin"}):
            with self.access():
                self.assertEqual(self.client().get("/api/state", headers=headers).status_code, 403)

    def test_proxy_requires_secret_peer_and_membership(self):
        secret = "synthetic-test-secret-" * 3
        config = dict(mode="auth_proxy", proxy_secret=secret, public_origin="https://docs.example.test", allowed_subjects=("alice",))
        valid = {"host": "docs.example.test", "x-docqa-proxy-secret": secret, "x-auth-request-user": "alice"}
        with self.access(**config):
            self.assertEqual(self.client().get("/api/state", headers=valid).status_code, 200)
            self.assertEqual(self.client().get("/api/state", headers={**valid, "x-docqa-proxy-secret": "forged"}).status_code, 401)
            self.assertEqual(self.client().get("/api/state", headers={**valid, "x-auth-request-user": "mallory"}).status_code, 403)
        with self.access(**config, proxy_peers=("192.0.2.1",)):
            self.assertEqual(self.client().get("/api/state", headers=valid).status_code, 403)

    def test_request_identity_is_server_owned(self):
        captured = []
        self.seed_workspace()
        self.server._graph = SimpleNamespace(invoke=lambda state: captured.append(state) or {"response": "ok"})
        with self.access():
            response = self.client().post("/api/chat", json={"query": "alpha"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured[0]["request_context"]["workspace_id"], "local")
        self.assertEqual(captured[0]["request_context"]["principal_id"], "local-developer")

    def test_missing_proxy_configuration_fails_closed(self):
        with self.access(mode="auth_proxy"):
            self.assertEqual(self.client().get("/api/state").status_code, 503)

    def test_response_headers_disable_framing_and_sniffing(self):
        response = self.client().get("/")
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(response.headers["x-frame-options"], "DENY")
        self.assertEqual(response.headers["cache-control"], "no-store")
