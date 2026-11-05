"""Step 24 bounded tool selection, execution, and chat integration."""

import json
import os
import re
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agents.orchestrator_agent import invoke_orchestrator as original_invoke_orchestrator
from tests.support import OfflineCase


class ToolLoopTests(OfflineCase):
    def test_ollama_tool_planner_requests_json_mode_without_changing_chat_model(self):
        from agents import orchestrator_agent
        model = Mock()
        bound = model.bind.return_value
        with patch.object(orchestrator_agent, "_get_ollama_settings",
                          return_value=("synthetic-model", "http://127.0.0.1:11434")), \
                patch.object(orchestrator_agent, "invoke_orchestrator",
                             new=original_invoke_orchestrator), \
                patch.object(orchestrator_agent, "_get_ollama_llm", return_value=model), \
                patch.object(orchestrator_agent, "_invoke_and_record",
                             return_value=SimpleNamespace(content='{"action":"final","answer":"done"}')) \
                             as invoke:
            orchestrator_agent.invoke_tool_planner("synthetic prompt", provider="ollama")
            model.bind.assert_called_once_with(format="json")
            self.assertIs(invoke.call_args.args[0], bound)
            model.bind.reset_mock()
            orchestrator_agent.invoke_orchestrator("ordinary prompt", provider="ollama")
            model.bind.assert_not_called()
            self.assertIs(invoke.call_args.args[0], model)

    def setup_turn(self, suffix=""):
        from runtime.events.repository import TurnEventRepository
        from storage.conversations import ConversationRepository
        root = self.temp / f"tool-loop{suffix}"
        conversations = ConversationRepository(root)
        reservation = conversations.reserve("alpha", "alice", None, "tool-loop-turn", "Calculate 0.1 + 0.2")
        return reservation, TurnEventRepository(root)

    @staticmethod
    def scripted_model(*decisions):
        answers = iter(decisions)
        prompts = []

        def invoke(prompt, *, provider, **kwargs):
            prompts.append(prompt)
            return SimpleNamespace(content=json.dumps(next(answers)))

        return invoke, prompts

    def test_policy_filtered_math_call_and_final_answer_have_visible_events(self):
        from capabilities.builtin import build_default_registry
        from runtime.tool_loop import ToolLoopRuntime
        reservation, events = self.setup_turn()
        model, prompts = self.scripted_model(
            {"action": "tool", "capability_id": "math.evaluate",
             "arguments": {"expression": "0.1 + 0.2"}},
            {"action": "final", "answer": "The result is 0.3."},
        )
        result = ToolLoopRuntime(build_default_registry(), events, model).run(
            "Calculate 0.1 + 0.2", "ollama", "alpha", "alice",
            reservation.conversation_id, "tool-loop-turn")
        self.assertEqual(result["answer"], "The result is 0.3.")
        self.assertEqual(result["tool_results"][0]["value"], "0.3")
        self.assertIn('"math.evaluate"', prompts[0])
        self.assertNotIn('"memory.save_preference"', prompts[0])
        self.assertIn('"0.3"', prompts[1])
        replay = events.replay("alpha", "alice", reservation.conversation_id)
        self.assertEqual([event["event_type"] for event in replay],
                         ["turn.started", "capability.selected", "tool.requested",
                          "tool.started", "tool.completed"])
        self.assertNotIn("0.1 + 0.2", json.dumps(replay))

    def test_malformed_and_disallowed_calls_fail_closed(self):
        from capabilities.builtin import build_default_registry
        from runtime.tool_loop import ToolLoopRejected, ToolLoopRuntime
        for index, (decision, reason) in enumerate((
            ({"action": "tool", "capability_id": "math.evaluate",
              "arguments": {"expression": "__import__('os').system('id')"}}, "invalid_arguments"),
            ({"action": "tool", "capability_id": "memory.save_preference",
              "arguments": {"key": "theme", "value": "dark"}}, "capability_not_allowed"),
            ({"action": "tool", "capability_id": "unknown.tool", "arguments": {}},
             "capability_not_allowed"),
        )):
            with self.subTest(reason=reason, decision=decision):
                reservation, events = self.setup_turn(str(index))
                model, _ = self.scripted_model(decision)
                with self.assertRaises(ToolLoopRejected) as raised:
                    ToolLoopRuntime(build_default_registry(), events, model).run(
                        "Do something", "ollama", "alpha", "alice",
                        reservation.conversation_id, "tool-loop-turn")
                self.assertEqual(raised.exception.reason, reason)
                self.assertEqual(events.replay("alpha", "alice", reservation.conversation_id)[-1]
                                 ["event_type"], "tool.failed")

    def test_call_budget_stops_repeated_model_requests(self):
        from capabilities.builtin import build_default_registry
        from runtime.tool_loop import ToolLoopRejected, ToolLoopRuntime
        reservation, events = self.setup_turn()
        decision = {"action": "tool", "capability_id": "math.evaluate",
                    "arguments": {"expression": "2 + 2"}}
        model, prompts = self.scripted_model(*(decision for _ in range(4)))
        with self.assertRaises(ToolLoopRejected) as raised:
            ToolLoopRuntime(build_default_registry(), events, model).run(
                "Calculate", "ollama", "alpha", "alice",
                reservation.conversation_id, "tool-loop-turn")
        self.assertEqual(raised.exception.reason, "call_budget_exhausted")
        self.assertEqual(len(prompts), 4)
        self.assertEqual(sum(event["event_type"] == "tool.completed" for event in events.replay(
            "alpha", "alice", reservation.conversation_id)), 3)

    def test_oversized_context_fails_before_model_call_with_audit_event(self):
        from capabilities.builtin import build_default_registry
        from runtime.tool_loop import ToolLoopRejected, ToolLoopRuntime
        reservation, events = self.setup_turn()
        def unexpected_model(_prompt, *, provider):
            self.fail("Model must not receive an oversized context")
        with self.assertRaises(ToolLoopRejected) as raised:
            ToolLoopRuntime(build_default_registry(), events, unexpected_model).run(
                "x" * 12001, "ollama", "alpha", "alice",
                reservation.conversation_id, "tool-loop-turn")
        self.assertEqual(raised.exception.reason, "context_budget_exhausted")
        self.assertEqual(events.replay("alpha", "alice", reservation.conversation_id)[-1]
                         ["event_type"], "tool.failed")

    def test_grounded_calculation_requires_numbers_in_current_evidence(self):
        from agents.grounded_tools import grounded_tools_agent
        from capabilities.builtin import build_default_registry
        from langchain_core.documents import Document
        from runtime.tool_loop import ToolLoopRuntime
        from storage.conversations import ConversationRepository
        from runtime.events.repository import TurnEventRepository
        root = self.temp / "grounded-tool"
        conversations = ConversationRepository(root)
        events = TurnEventRepository(root)
        for index, (expression, expected) in enumerate((("30 + 15", "45"),
                                                         ("30 + 999", None))):
            turn_id = f"grounded-{index}"
            turn = conversations.reserve("alpha", "alice", None, turn_id, "Add the deadlines")
            model, prompts = self.scripted_model(
                {"action": "tool", "capability_id": "math.evaluate",
                 "arguments": {"expression": expression}},
                {"action": "final", "answer": "Done"},
            )
            state = {"query": "Add the deadlines", "provider": "ollama",
                     "documents": [Document(page_content="Alpha deadline: 30 days. Beta: 15 days.",
                                            metadata={"source": "policy.txt"})],
                     "request_context": {"workspace_id": "alpha", "principal_id": "alice",
                                         "_turn_attempt": turn.lease_token},
                     "conversation_id": turn.conversation_id, "turn_id": turn_id,
                     "grounded_tool_runtime": ToolLoopRuntime(build_default_registry(), events, model)}
            result = grounded_tools_agent(state)
            self.assertNotIn('"memory.save_preference"', prompts[0])
            self.assertNotIn('"time.current_utc"', prompts[0])
            self.assertEqual(len(prompts), 1)
            if expected is None:
                self.assertEqual(result["builtin_tool_status"]["reason"], "invalid_arguments")
                self.assertNotIn("builtin_tool_results", result)
            else:
                self.assertEqual(result["builtin_tool_results"][0]["value"], expected)
                self.assertEqual(result["builtin_tool_results"][0]["evidence_ids"],
                                 [result["documents"][0].metadata["evidence_id"]])

    def test_grounded_summary_does_not_invoke_calculator(self):
        from agents.grounded_tools import grounded_tools_agent
        from langchain_core.documents import Document
        state = {"query": "Summarize the deadline policy",
                 "documents": [Document(page_content="Alpha: 30 days. Beta: 15 days.",
                                        metadata={"source": "policy.txt"})],
                 "grounded_tool_runtime": object()}
        self.assertIs(grounded_tools_agent(state), state)
        self.assertNotIn("builtin_tool_results", state)

    def test_calculation_answer_rejects_citation_to_wrong_source(self):
        from agents.evidence import finalize_answer, prepare_evidence
        from langchain_core.documents import Document
        state = {"documents": [
            Document(page_content="Alpha: 30 days. Beta: 15 days.",
                     metadata={"source": "deadlines.txt"}),
            Document(page_content="Unrelated status from another source.",
                     metadata={"source": "status.txt"}),
        ]}
        prepare_evidence(state)
        first, second = [item.metadata["evidence_id"] for item in state["documents"]]
        state["builtin_tool_results"] = [{"capability_id": "math.evaluate", "value": "45",
                                          "evidence_ids": [first]}]
        state["response"] = f"Combined deadline is 45 days [{second}]."
        finalized = finalize_answer(state)
        self.assertEqual(finalized["result_status"], "citation_error")
        self.assertEqual(finalized["citations"], [])
        self.assertNotIn("45 days", finalized["response"])

    def test_grounded_document_chat_uses_validated_tool_result_and_citations(self):
        from langchain_core.documents import Document
        from langgraph_flow.graph_builder import build_graph
        self.seed_workspace([Document(page_content="Alpha deadline is 30 days and beta is 15 days.",
                                      metadata={"source": "policy.txt"})])
        self.server._graph = build_graph()
        self.stub_graph_models()
        model, _ = self.scripted_model(
            {"action": "tool", "capability_id": "math.evaluate",
             "arguments": {"expression": "30 + 15"}},
            {"action": "final", "answer": "Untrusted planner final"},
        )
        qa_prompts = []
        def answer_from_evidence(prompt, *, provider):
            qa_prompts.append(prompt)
            identifier = re.search(r'E-[a-f0-9]{12}', prompt).group(0)
            return SimpleNamespace(content=f"The combined deadline is 45 days [{identifier}].")
        with patch.object(self.server.orchestrator_agent, "invoke_orchestrator", side_effect=model), \
                patch("agents.qa_agent.invoke_orchestrator", side_effect=answer_from_evidence):
            response = self.client().post("/api/chat", json={
                "query": "Add the alpha and beta deadlines", "use_tools": True,
                "idempotency_key": "grounded-chat-0001"})
        self.assertEqual(response.status_code, 200, response.text)
        answer = response.json()
        self.assertEqual(answer["result_status"], "answered")
        self.assertEqual(answer["tool_result"][0]["value"], "45")
        self.assertEqual(answer["tool_result"][0]["evidence_ids"],
                         [answer["citations"][0]["evidence_id"]])
        self.assertIn('"value": "45"', qa_prompts[0])
        self.assertNotIn("Untrusted planner final", answer["response"])
        self.assertTrue(answer["documents"])

    def test_grounded_web_chat_keeps_web_origin_and_citations(self):
        from langchain_core.documents import Document
        from langgraph_flow.graph_builder import build_graph
        def synthetic_search(state):
            state["documents"] = [Document(
                page_content="Report A: 12 cases. Report B: 8 cases.",
                metadata={"source": "https://example.test/report", "content_type": "web_result",
                          "url": "https://example.test/report"})]
            state["source"] = "Web"
            return state
        with patch("langgraph_flow.graph_builder.web_search_agent", side_effect=synthetic_search):
            self.server._graph = build_graph()
        model, _ = self.scripted_model(
            {"action": "tool", "capability_id": "math.evaluate",
             "arguments": {"expression": "12 + 8"}},
            {"action": "final", "answer": "Ignore web citations"},
        )
        def web_answer(prompt, *, provider):
            identifier = re.search(r'E-[a-f0-9]{12}', prompt).group(0)
            return SimpleNamespace(content=f"The reports show 20 cases [{identifier}].")
        with patch.object(self.server.orchestrator_agent, "invoke_orchestrator", side_effect=model), \
                patch("agents.qa_agent.invoke_orchestrator", side_effect=web_answer):
            response = self.client().post("/api/chat", json={
                "query": "Add the cases", "use_tools": True, "use_web_search": True,
                "idempotency_key": "grounded-web-0001"})
        self.assertEqual(response.status_code, 200, response.text)
        answer = response.json()
        self.assertEqual(answer["tool_result"][0]["value"], "20")
        self.assertEqual(answer["citations"][0]["origin"], "web")
        self.assertEqual(answer["tool_result"][0]["evidence_ids"],
                         [answer["citations"][0]["evidence_id"]])
        self.assertNotIn("Ignore web citations", answer["response"])

    def test_grounded_connector_tool_result_is_redacted_after_revocation(self):
        from langgraph_flow.graph_builder import build_graph
        self.stub_graph_models()
        self.server._graph = build_graph()
        source_root = self.temp / "grounded-connector"
        (source_root / "policies").mkdir(parents=True)
        (source_root / "policies" / "deadlines.txt").write_text(
            "Alpha deadline is 30 days. Beta deadline is 15 days.")
        model, _ = self.scripted_model(
            {"action": "tool", "capability_id": "math.evaluate",
             "arguments": {"expression": "30 + 15"}},
            {"action": "final", "answer": "Done"},
        )
        with patch.dict(os.environ, {"DOCQA_CONNECTOR_ROOT": str(source_root),
                                     "DOCQA_CONNECTOR_SOURCE_ACL": "{}"}), \
                patch.object(self.server.orchestrator_agent, "invoke_orchestrator", side_effect=model):
            client = self.client()
            created = client.post("/api/connectors/sources", json={
                "source_key": "policies", "display_name": "Policies",
                "idempotency_key": "grounded-create-0001"})
            self.assertEqual(created.status_code, 201, created.text)
            source = created.json()["source"]
            path = f"/api/connectors/sources/{source['source_id']}"
            self.assertEqual(client.post(path + "/sync", headers={
                "idempotency-key": "grounded-sync-0001"}).status_code, 200)
            self.assertEqual(client.post(path + "/publish").status_code, 200)
            answer = client.post("/api/chat", json={
                "query": "Add the alpha and beta deadlines", "use_tools": True,
                "idempotency_key": "grounded-connector-turn"})
            self.assertEqual(answer.status_code, 200, answer.text)
            body = answer.json()
            self.assertEqual(body["tool_result"][0]["value"], "45")
            self.assertEqual(body["connector_source_ids"], [source["source_id"]])
            self.assertEqual(body["citations"][0]["origin"], "connector")
            self.assertEqual(client.delete(f"/api/connectors/grants/{source['grant_id']}").status_code,
                             200)
            redacted = client.get(f"/api/conversations/{body['conversation_id']}").json()
            response = redacted["messages"][-1]["response"]
            self.assertEqual(response["result_status"], "source_revoked")
            self.assertIsNone(response["tool_result"])
            self.assertNotIn("45 days", str(redacted))

    def test_chat_opt_in_uses_tool_loop_and_default_chat_is_unchanged(self):
        from app import server
        self.server._graph = SimpleNamespace(invoke=lambda state: state)
        model, _ = self.scripted_model(
            {"action": "tool", "capability_id": "math.evaluate",
             "arguments": {"expression": "0.1 + 0.2"}},
            {"action": "final", "answer": "0.3"},
        )
        with patch.object(server.orchestrator_agent, "invoke_orchestrator", side_effect=model):
            client = self.client()
            response = client.post("/api/chat", json={
                "query": "Calculate 0.1 + 0.2", "use_tools": True,
                "idempotency_key": "tool-chat-turn"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["response"], "0.3")
        self.assertEqual(response.json()["tool_result"][0]["value"], "0.3")
        identifier = response.json()["conversation_id"]
        replay = client.get(f"/api/conversations/{identifier}/events").json()["events"]
        self.assertIn("tool.completed", [event["event_type"] for event in replay])
        with patch.object(server.orchestrator_agent, "invoke_orchestrator",
                          return_value=SimpleNamespace(content="Ordinary reply")):
            ordinary = client.post("/api/chat", json={
                "query": "Hello", "idempotency_key": "ordinary-chat-turn"})
        self.assertEqual(ordinary.status_code, 200, ordinary.text)
        self.assertEqual(ordinary.json()["response"], "Ordinary reply")
        ordinary_events = client.get(
            f"/api/conversations/{ordinary.json()['conversation_id']}/events"
        ).json()["events"]
        self.assertNotIn("tool.requested", [event["event_type"] for event in ordinary_events])

    def test_write_proposal_needs_explicit_owner_approval_and_is_idempotent(self):
        from app import server
        from storage.tool_approvals import ToolApprovalNotFound, ToolApprovalRepository
        self.server._graph = SimpleNamespace(invoke=lambda state: state)
        model, prompts = self.scripted_model(
            {"action": "tool", "capability_id": "memory.save_preference",
             "arguments": {"key": "Answer style", "value": "Concise bullets"}},
        )
        with patch.object(server.orchestrator_agent, "invoke_orchestrator", side_effect=model):
            client = self.client()
            proposed = client.post("/api/chat", json={
                "query": "Remember that I prefer concise bullets", "use_tools": True,
                "idempotency_key": "approved-memory-turn"})
        self.assertEqual(proposed.status_code, 200, proposed.text)
        self.assertIn('"memory.save_preference"', prompts[0])
        self.assertEqual(proposed.json()["result_status"], "approval_required")
        approval = proposed.json()["approval"]
        self.assertEqual(approval["status"], "pending")
        self.assertEqual(client.get("/api/memories/semantic").json()["memories"], [])
        root = self.temp / "vector_db"
        with self.assertRaises(ToolApprovalNotFound):
            ToolApprovalRepository(root).get("local", "someone-else", approval["approval_id"])
        identifier = approval["approval_id"]
        decision = client.post(f"/api/tool-approvals/{identifier}/decision",
                               json={"decision": "approve"})
        self.assertEqual(decision.status_code, 200, decision.text)
        repeated = client.post(f"/api/tool-approvals/{identifier}/decision",
                               json={"decision": "approve"})
        self.assertEqual(repeated.json(), decision.json())
        self.assertEqual(client.post(f"/api/tool-approvals/{identifier}/decision",
                                     json={"decision": "reject"}).status_code, 409)
        memories = client.get("/api/memories/semantic").json()["memories"]
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["value"], "Concise bullets")
        conversation = client.get(f"/api/conversations/{proposed.json()['conversation_id']}").json()
        self.assertEqual(conversation["messages"][-1]["response"]["approval"]["status"], "approved")
        events = client.get(f"/api/conversations/{proposed.json()['conversation_id']}/events").json()["events"]
        self.assertIn("approval.required", [item["event_type"] for item in events])
        self.assertIn("approval.resolved", [item["event_type"] for item in events])
        self.assertIn("tool.completed", [item["event_type"] for item in events])
        self.assertNotIn("Concise bullets", json.dumps(events))

    def test_secret_write_proposal_is_rejected_before_approval(self):
        from app import server
        self.server._graph = SimpleNamespace(invoke=lambda state: state)
        model, _ = self.scripted_model(
            {"action": "tool", "capability_id": "memory.save_preference",
             "arguments": {"key": "API key", "value": "sk-abcdefghijklmnopqrstuvwxyz"}},
        )
        with patch.object(server.orchestrator_agent, "invoke_orchestrator", side_effect=model):
            client = self.client()
            result = client.post("/api/chat", json={
                "query": "Remember a secret", "use_tools": True,
                "idempotency_key": "secret-memory-turn"})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["result_status"], "tool_error")
        self.assertIsNone(result.json()["approval"])
        self.assertEqual(client.get("/api/memories/semantic").json()["memories"], [])

    def test_reject_and_expire_never_write_and_deleted_turn_removes_approval(self):
        from storage.conversations import ConversationRepository
        from storage.tool_approvals import ToolApprovalNotFound, ToolApprovalRepository
        root = self.temp / "approval-lifecycle"
        conversations = ConversationRepository(root)
        approvals = ToolApprovalRepository(root)
        for suffix, decision, clock_offset in (("reject", "reject", 1),
                                               ("expire", "approve", approvals.RETENTION_SECONDS)):
            turn = conversations.reserve("alpha", "alice", None, f"turn-{suffix}", "Remember style")
            import time
            current = int(time.time())
            proposal = approvals.propose_preference("alpha", "alice", turn.conversation_id,
                f"turn-{suffix}", turn.lease_token, "Style", suffix, now=current)
            conversations.complete("alpha", "alice", turn.conversation_id, f"turn-{suffix}",
                turn.lease_token, {"response": "Review preference", "approval": proposal,
                                   "result_status": "approval_required"})
            resolved = approvals.resolve("alpha", "alice", proposal["approval_id"],
                                         decision, now=current + clock_offset)
            self.assertEqual(resolved["status"], "rejected" if suffix == "reject" else "expired")
            self.assertEqual(approvals.semantic.recall("alpha", "alice"), [])
            conversations.delete("alpha", "alice", turn.conversation_id)
            with self.assertRaises(ToolApprovalNotFound):
                approvals.get("alpha", "alice", proposal["approval_id"])

    def test_memory_capacity_failure_leaves_approval_pending(self):
        from memory.semantic_memory import SemanticMemoryRepository
        from storage.conversations import ConversationRepository
        from storage.tool_approvals import ToolApprovalConflict, ToolApprovalRepository
        root = self.temp / "approval-capacity"
        conversations = ConversationRepository(root)
        approvals = ToolApprovalRepository(root)
        turn = conversations.reserve("alpha", "alice", None, "capacity-turn", "Remember style")
        proposal = approvals.propose_preference("alpha", "alice", turn.conversation_id,
            "capacity-turn", turn.lease_token, "Style", "Compact")
        conversations.complete("alpha", "alice", turn.conversation_id, "capacity-turn",
            turn.lease_token, {"response": "Review preference", "approval": proposal,
                               "result_status": "approval_required"})
        semantic = SemanticMemoryRepository(root)
        for index in range(semantic.MAX_ACTIVE):
            semantic.upsert("alpha", "alice", "preference", f"Other {index}", "Stored")
        with self.assertRaises(ToolApprovalConflict):
            approvals.resolve("alpha", "alice", proposal["approval_id"], "approve")
        self.assertEqual(approvals.get("alpha", "alice", proposal["approval_id"])["status"],
                         "pending")
        self.assertEqual(len(semantic.list_recent("alpha", "alice")), semantic.MAX_ACTIVE)

    def test_calculator_rejects_code_and_unbounded_arithmetic(self):
        from capabilities.read_tools import InvalidToolArguments, evaluate_expression
        self.assertEqual(evaluate_expression({"expression": "0.1 + 0.2"})["value"], "0.3")
        for expression in ("__import__('os').system('id')", "2 ** 100", "1 / 0",
                           "(1 + 2) * True", "10 ** 6 ** 6"):
            with self.subTest(expression=expression), self.assertRaises(InvalidToolArguments):
                evaluate_expression({"expression": expression})

    def test_stop_discards_uncommitted_write_proposal(self):
        from runtime.events.repository import TurnEventRepository
        from storage.conversations import ConversationRepository
        from storage.tool_approvals import ToolApprovalNotFound, ToolApprovalRepository
        root = self.temp / "approval-cancel"
        conversations = ConversationRepository(root)
        events = TurnEventRepository(root)
        approvals = ToolApprovalRepository(root)
        turn = conversations.reserve("alpha", "alice", None, "approval-stop", "Remember style")
        proposal = approvals.propose_preference("alpha", "alice", turn.conversation_id,
            "approval-stop", turn.lease_token, "Style", "Compact")
        events.request_cancel("alpha", "alice", turn.conversation_id, "approval-stop")
        with self.assertRaises(ToolApprovalNotFound):
            approvals.get("alpha", "alice", proposal["approval_id"])
        self.assertEqual(approvals.semantic.recall("alpha", "alice"), [])


if __name__ == "__main__":
    import unittest
    unittest.main()
