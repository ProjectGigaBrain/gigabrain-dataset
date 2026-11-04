"""Testes do realizador ``llm`` (variante B) com provedores falsos: nenhum
teste chama LLM de verdade."""

import json
import random

import pytest

from pureprep.dialogue import build_meetings, profiles
from pureprep.dialogue import realize_llm as RL
from pureprep.dialogue.llm_providers import DryRunProvider, LLMFatal, LLMUnavailable
from pureprep.dialogue.model import Phenomenon, Turn, Utt
from tests.test_meetings import _check_invariants, portal  # noqa: F401  (fixture)


class EchoProvider:
    """LLM falso "bem-comportado": devolve o conteúdo com um marcador de
    paráfrase e respostas curtas para os atos sem conteúdo."""

    name = "fake"

    def __init__(self, mutate=None):
        self.calls = 0
        self.mutate = mutate or (lambda item, text: text)

    def params(self):
        return {"provider": self.name, "model": "echo-1"}

    def complete(self, system, user, schema):
        self.calls += 1
        payload = json.loads(user)
        out = []
        for item in payload["turns"]:
            if item.get("content"):
                text = "PARAPHRASE " + " ".join(c["text"] for c in item["content"])
            else:
                text = "Okay." if item["act"] in RL._SHORT_ACTS else "Alright, moving on."
            out.append({"id": item["id"], "text": self.mutate(item, text)})
        return json.dumps({"turns": out})


def _run(portal, provider, tmp_path, profile="stress", seed=2, **kw):
    doc, ext, reqs = portal
    return build_meetings(doc, ext, reqs, profile=profile, realizer="llm", seed=seed,
                          realizer_options={"provider": provider, "cache_dir": tmp_path / "cache", "workers": 4, **kw})


def test_same_plan_and_gold_as_variant_a(portal, tmp_path):
    doc, ext, reqs = portal
    _, _, ga = build_meetings(doc, ext, reqs, profile="stress", realizer="rules", seed=2)
    plan, realized, gb = _run(portal, EchoProvider(), tmp_path)
    _check_invariants(plan, realized, gb)
    for key in ("requirements", "issues", "distractors"):
        a = [{k: v for k, v in x.items() if k != "trace"} for x in ga[key]]
        b = [{k: v for k, v in x.items() if k != "trace"} for x in gb[key]]
        assert a == b, key
    stats = gb["generator"]["realizer_stats"]
    assert stats["llm_accepted"] > 0.8 * stats["turns"]
    assert gb["generator"]["params"]["llm"]["model"] == "echo-1"
    assert "PARAPHRASE" in "".join(t for _, t, _ in realized)


def test_cache_makes_b_reproducible_offline(portal, tmp_path):
    first = EchoProvider()
    _, r1, g1 = _run(portal, first, tmp_path)
    assert first.calls == g1["_run_stats"]["requests"] == g1["generator"]["realizer_stats"]["windows"]
    second = EchoProvider()
    _, r2, g2 = _run(portal, second, tmp_path, offline=True)
    assert second.calls == 0
    assert [t for _, t, _ in r1] == [t for _, t, _ in r2]
    assert g2["_run_stats"]["cache_hits"] == g2["generator"]["realizer_stats"]["windows"]
    # o gabarito não registra como a execução rodou: original e reprodução são idênticos
    g1.pop("_run_stats"), g2.pop("_run_stats")
    assert g1 == g2


def test_offline_without_cache_fails(portal, tmp_path):
    with pytest.raises(RuntimeError, match="offline"):
        _run(portal, EchoProvider(), tmp_path / "vazio", offline=True)


def test_dry_run_falls_back_to_rules_everywhere(portal, tmp_path):
    doc, ext, reqs = portal
    _, ra, _ = build_meetings(doc, ext, reqs, profile="stress", realizer="rules", seed=2)
    dry = DryRunProvider()
    _, rb, gb = _run(portal, dry, tmp_path)
    assert [t for _, t, _ in ra] == [t for _, t, _ in rb]  # B sem falas aceitas == A
    assert dry.estimate()["requests"] == gb["generator"]["realizer_stats"]["windows"]
    assert gb["generator"]["realizer_stats"].get("llm_accepted", 0) == 0


def test_bad_outputs_are_rejected_and_fall_back(portal, tmp_path):
    def inventor(item, text):  # inventa um número em toda fala com conteúdo
        return text + " within 999 seconds" if item.get("content") else text
    _, _, g = _run(portal, EchoProvider(inventor), tmp_path)
    st = g["generator"]["realizer_stats"]
    assert st.get("fallback:invented_number", 0) > 0
    assert st.get("llm_accepted_content", 0) == 0


def test_prompts_never_reveal_removed_details(portal, tmp_path):
    doc, ext, reqs = portal
    from pureprep.dialogue.plan import build_plan
    plan = build_plan(doc, ext, reqs, "stress", random.Random(f"2:{doc.doc_id}:meetings:stress"))
    removed = {p.pid: p.details.get("removed") or p.details.get("missing") for p in plan.phenomena
               if p.type in ("ambiguity", "incompleteness") and (p.details.get("removed") or p.details.get("missing"))}
    assert removed
    checked = 0
    for m in plan.meetings:
        for i, turn in enumerate(m.turns, 1):
            sent = RL._turn_payload(turn, f"{m.mid}:T{i}").get("content", [])
            if turn.act != "context":
                assert [c["text"] for c in sent] == [u.text for u in turn.utts]  # o prompt leva só o conteúdo do plano
            for u in turn.utts:
                if u.pid in removed and u.role in ("statement", "part1", "part2"):
                    checked += 1
                    span = removed[u.pid]
                    # o trecho removido não vai no prompt (cortes longos podem repetir palavras soltas)
                    assert span not in u.text, (u.pid, span)
    assert checked


# ----------------------------------------------------------------- verify()


def _phen(**d):
    return {"P1": Phenomenon("P1", d.pop("type"), "R1", d.pop("kind"), details=d)}


def test_verify_numbers_and_content():
    t = Turn("Client", "state", [Utt("R1", "Search results shall appear in less than 2 seconds.", "statement")])
    assert RL.verify(t, "So search results have to show up in under 2 seconds.", {}) is None
    assert RL.verify(t, "Search results have to show up really fast.", {}) == "lost_number"
    assert RL.verify(t, "Search results have to show up in 2 seconds, 5 at most.", {}) == "invented_number"
    assert RL.verify(t, "We want 2 of those.", {}) == "lost_content"


def test_verify_blocks_leaked_detail_and_mixed_values():
    amb = Turn("Client", "state", [Utt("R1", "Someone shall approve the order.", "statement", "P1")])
    phen = _phen(type="ambiguity", kind="vague_actor", removed="The supervisor", replacement="someone")
    assert RL.verify(amb, "Someone has to approve the order.", phen) is None
    assert RL.verify(amb, "The supervisor has to approve the order.", phen) == "leaked_detail"
    dis = Turn("TechLead", "counter", [Utt("R1", "The warning shall be given 15 sec before.", "distractor", "P1")])
    phen = _phen(type="conflict", kind="value", original_value="5 sec", proposed_value="15 sec")
    assert RL.verify(dis, "Hmm, I'd say the warning comes 15 sec before.", phen) is None
    assert RL.verify(dis, "I'd say the warning comes 5 sec before.", phen) in ("invented_number", "lost_number")
    assert RL.verify(dis, "Either 5 sec or 15 sec before, I'd say the warning comes.", phen) is not None


def test_verify_short_acts_and_cleanup():
    t = Turn("Analyst", "backchannel")
    assert RL.verify(t, "Mm-hmm.", {}) is None
    assert RL.verify(t, "Right, that makes a lot of sense to me, thanks.", {}) == "too_long"
    assert RL.clean_text('Analyst: "Okay, got it."', "Analyst") == "Okay, got it."


def test_provider_failure_is_counted(portal, tmp_path):
    class Flaky(EchoProvider):  # falha nas janelas que começam no turno 1 de cada reunião
        def complete(self, system, user, schema):
            if json.loads(user)["turns"][0]["id"].endswith(":T1"):
                raise LLMUnavailable("HTTP 529: overloaded")
            return super().complete(system, user, schema)
    _, _, g = _run(portal, Flaky(), tmp_path)
    st = g["_run_stats"]
    assert 0 < st["request_failed:HTTP 529: overloaded"] < st["windows"]
    assert not any(k.startswith("request_failed") for k in g["generator"]["realizer_stats"])
    assert st["fallback:no_response"] > 0 and st["llm_accepted"] > 0


def test_all_requests_failing_stops_instead_of_copying_a(portal, tmp_path):
    class Broken(EchoProvider):
        def complete(self, system, user, schema):
            raise LLMUnavailable("conexão: recusada")
    with pytest.raises(LLMFatal, match="conexão: recusada"):
        _run(portal, Broken(), tmp_path)


@pytest.fixture(autouse=True)
def _stress_profile(monkeypatch):
    monkeypatch.setitem(profiles.PROFILES, "stress", {**profiles.PROFILES["hard"], "meetings": (3, 3), "p_omit": 0.06,
                                                       "p_conflict": 0.2, "p_ambiguity": 0.15, "p_inconsistency": 0.15,
                                                       "p_incomplete": 0.15, "p_resolve_later": 0.6, "p_complete_later": 0.6})


# ----------------------------------------------------------------- provedor Anthropic (servidor falso local)


@pytest.fixture
def fake_api(monkeypatch):
    """Servidor HTTP local que imita a Messages API; nada sai da máquina."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    state = {"status": 200, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            state["requests"].append({"headers": dict(self.headers), "body": body})
            if state["status"] == 200:
                resp = {"id": "msg_x", "type": "message", "role": "assistant", "model": body["model"],
                        "content": [{"type": "text", "text": '{"turns": []}'}], "stop_reason": "end_turn",
                        "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}}
            else:
                resp = {"type": "error", "error": {"type": "invalid_request_error", "message": "not scoped to a workspace"}}
            data = json.dumps(resp).encode()
            self.send_response(state["status"])
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{srv.server_port}")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-teste-local")
    yield state
    srv.shutdown()


def test_anthropic_provider_sends_workspace_header(fake_api):
    pytest.importorskip("anthropic")
    from pureprep.dialogue.llm_providers import AnthropicProvider
    AnthropicProvider(workspace_id="wrkspc_teste").complete(RL.SYSTEM, "{}", RL.SCHEMA)
    req = fake_api["requests"][-1]
    assert req["headers"]["anthropic-workspace-id"] == "wrkspc_teste"
    assert req["body"]["model"] == "claude-opus-5-5" and req["body"]["fallbacks"] == "default"


def test_fatal_api_error_stops_generation(fake_api, portal, tmp_path):
    pytest.importorskip("anthropic")
    from pureprep.dialogue.llm_providers import AnthropicProvider, LLMFatal
    fake_api["status"] = 400
    with pytest.raises(LLMFatal, match="HTTP 400"):
        _run(portal, AnthropicProvider(max_retries=0), tmp_path)


# ----------------------------------------------------------------- provedores HTTP locais (ollama, openai)


@pytest.fixture
def local_server():
    """Servidor HTTP local configurável: ``routes[(método, caminho)] = (status, corpo)``
    ou uma função ``body -> (status, corpo)``. Guarda as requisições recebidas."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    state = {"routes": {}, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, method):
            n = int(self.headers.get("content-length") or 0)
            body = json.loads(self.rfile.read(n)) if n else None
            state["requests"].append({"method": method, "path": self.path, "headers": dict(self.headers), "body": body})
            route = state["routes"].get((method, self.path), (404, {"error": "not found"}))
            status, resp = route(body) if callable(route) else route
            data = json.dumps(resp).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._reply("GET")

        def do_POST(self):
            self._reply("POST")

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{srv.server_port}"
    yield state
    srv.shutdown()


def _free_port_url():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{s.getsockname()[1]}"


def _ollama(local_server, **kw):
    from pureprep.dialogue.llm_providers import OllamaProvider
    return OllamaProvider(kw.pop("model", "qwen3:14b"), base_url=local_server["url"] + "/v1", max_retries=0, **kw)


def test_ollama_sends_schema_and_num_ctx_and_strips_thinking(local_server):
    local_server["routes"][("POST", "/api/chat")] = (200, {
        "message": {"role": "assistant", "content": '<think>hmm {x}</think>{"turns": []}'}, "done_reason": "stop"})
    p = _ollama(local_server, num_ctx=12000)
    assert p.complete(RL.SYSTEM, "{}", RL.SCHEMA) == '{"turns": []}'
    req = local_server["requests"][-1]
    assert req["path"] == "/api/chat"  # o /v1 do endereço compatível é descartado
    assert req["body"]["format"] == RL.SCHEMA and req["body"]["stream"] is False
    assert req["body"]["options"]["num_ctx"] == 12000
    assert "authorization" not in {k.lower() for k in req["headers"]}
    assert p.params() == {"provider": "ollama", "model": "qwen3:14b"}  # chave do cache não depende do endereço


def test_ollama_check_reports_missing_model_and_server_down(local_server):
    local_server["routes"][("GET", "/api/tags")] = (200, {"models": [{"name": "gemma3:12b"}, {"name": "llama3.1:latest"}]})
    _ollama(local_server, model="gemma3:12b").check()
    _ollama(local_server, model="llama3.1").check()  # sem tag = :latest
    with pytest.raises(LLMFatal, match="ollama pull qwen3:14b"):
        _ollama(local_server).check()
    from pureprep.dialogue.llm_providers import OllamaProvider
    with pytest.raises(LLMFatal, match="Ollama"):
        OllamaProvider("qwen3:14b", base_url=_free_port_url()).check()


def test_ollama_truncation_and_fatal_errors(local_server):
    local_server["routes"][("POST", "/api/chat")] = (200, {"message": {"content": '{"turns": ['}, "done_reason": "length"})
    with pytest.raises(LLMUnavailable, match="num-ctx"):
        _ollama(local_server).complete(RL.SYSTEM, "{}", RL.SCHEMA)
    local_server["routes"][("POST", "/api/chat")] = (404, {"error": "model 'qwen3:14b' not found"})
    with pytest.raises(LLMFatal, match="not found"):
        _ollama(local_server).complete(RL.SYSTEM, "{}", RL.SCHEMA)
    local_server["routes"][("POST", "/api/chat")] = (500, {"error": "out of memory"})
    with pytest.raises(LLMUnavailable, match="out of memory"):
        _ollama(local_server).complete(RL.SYSTEM, "{}", RL.SCHEMA)


def test_ollama_end_to_end_through_realizer(local_server, portal, tmp_path):
    def chat(body):
        reply = EchoProvider().complete(None, body["messages"][1]["content"], None)
        return 200, {"message": {"content": reply}, "done_reason": "stop"}
    local_server["routes"][("POST", "/api/chat")] = chat
    _, realized, g = _run(portal, _ollama(local_server), tmp_path)
    assert g["generator"]["realizer_stats"]["llm_accepted"] > 0.8 * g["generator"]["realizer_stats"]["turns"]
    assert g["generator"]["params"]["llm"]["provider"] == "ollama"
    assert (tmp_path / "cache" / "ollama-qwen3_14b").is_dir()


def _openai(local_server, **kw):
    from pureprep.dialogue.llm_providers import OpenAICompatibleProvider
    return OpenAICompatibleProvider("local-model", base_url=local_server["url"] + "/v1", max_retries=0, **kw)


def test_openai_compatible_works_without_sdk_or_key(local_server, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    local_server["routes"][("POST", "/v1/chat/completions")] = (200, {
        "choices": [{"message": {"content": '{"turns": []}'}, "finish_reason": "stop"}]})
    assert _openai(local_server).complete(RL.SYSTEM, "{}", RL.SCHEMA) == '{"turns": []}'
    req = local_server["requests"][-1]
    assert req["body"]["response_format"]["type"] == "json_schema"
    assert req["body"]["response_format"]["json_schema"]["schema"] == RL.SCHEMA
    assert "authorization" not in {k.lower() for k in req["headers"]}
    _openai(local_server, api_key="sk-x").complete(RL.SYSTEM, "{}", RL.SCHEMA)
    assert local_server["requests"][-1]["headers"]["Authorization"] == "Bearer sk-x"


def test_openai_compatible_falls_back_to_json_object(local_server):
    def chat(body):
        if body["response_format"]["type"] == "json_schema":
            return 400, {"error": {"message": "'response_format.type' must be 'json_object' or 'text'"}}
        return 200, {"choices": [{"message": {"content": '{"turns": []}'}, "finish_reason": "stop"}]}
    local_server["routes"][("POST", "/v1/chat/completions")] = chat
    p = _openai(local_server)
    assert p.complete(RL.SYSTEM, "{}", RL.SCHEMA) == '{"turns": []}'
    p.complete(RL.SYSTEM, "{}", RL.SCHEMA)
    types = [r["body"]["response_format"]["type"] for r in local_server["requests"]]
    assert types == ["json_schema", "json_object", "json_object"]  # só tenta o esquema uma vez


def test_openai_compatible_errors(local_server):
    local_server["routes"][("POST", "/v1/chat/completions")] = (200, {
        "choices": [{"message": {"content": '{"tu'}, "finish_reason": "length"}]})
    with pytest.raises(LLMUnavailable, match="truncada"):
        _openai(local_server).complete(RL.SYSTEM, "{}", RL.SCHEMA)
    local_server["routes"][("POST", "/v1/chat/completions")] = (401, {"error": {"message": "invalid api key"}})
    with pytest.raises(LLMFatal, match="invalid api key"):
        _openai(local_server).complete(RL.SYSTEM, "{}", RL.SCHEMA)
    from pureprep.dialogue.llm_providers import OpenAICompatibleProvider
    with pytest.raises(LLMFatal, match="rodando"):
        OpenAICompatibleProvider("m", base_url=_free_port_url() + "/v1").check()


def test_transient_errors_are_retried(local_server):
    from pureprep.dialogue.llm_providers import OllamaProvider
    calls = []

    def chat(body):
        calls.append(1)
        if len(calls) < 3:
            return 503, {"error": "loading model"}
        return 200, {"message": {"content": '{"turns": []}'}, "done_reason": "stop"}
    local_server["routes"][("POST", "/api/chat")] = chat
    p = OllamaProvider("qwen3:14b", base_url=local_server["url"], max_retries=3, retry_wait=0.01)
    assert p.complete(RL.SYSTEM, "{}", RL.SCHEMA) == '{"turns": []}'
    assert len(calls) == 3


def test_cli_stops_before_generating_when_ollama_is_down(tmp_path):
    from pureprep import cli
    with pytest.raises(LLMFatal, match="Ollama"):
        cli.main(["--realizer", "llm", "--llm-provider", "ollama", "--llm-model", "qwen3:14b",
                  "--llm-base-url", _free_port_url(), "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()
