"""End-to-end plumbing of OllamaLLM (LangChain ChatOllama + structured output) against a fake Ollama HTTP server."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from src.llm import OllamaLLM
from src.schemas import Proposal

GOOD = Proposal(action="OFFER", offer_id="TECH_SUPPORT_TRIAL", discount_pct=0, duration_months=3, channel="sms",
                message="Hello, we would like to add Tech Support at no charge for 3 months. Reply YES to accept. "
                        "Reply STOP to opt out.", rationale="fits", citations=["offer_catalog#tech-support-trial"])


def serve(content: str, seen: list):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):                                               # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            base = {"model": body["model"], "created_at": "2026-01-01T00:00:00Z"}
            lines = [{**base, "message": {"role": "assistant", "content": content}, "done": False},
                     {**base, "message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop",
                      "total_duration": 5_000_000, "load_duration": 1_000, "prompt_eval_count": 321,
                      "prompt_eval_duration": 1_000, "eval_count": 45, "eval_duration": 2_000}]
            payload = ("\n".join(json.dumps(x) for x in lines) + "\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture()
def fake_ollama():
    servers = []

    def start(content):
        seen = []
        srv = serve(content, seen)
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}", seen

    yield start
    for s in servers:
        s.shutdown()


def test_valid_json_is_parsed_into_a_proposal_with_token_counts(fake_ollama):
    url, seen = fake_ollama(GOOD.model_dump_json())
    res = OllamaLLM(model="granite4:micro", base_url=url).draft([("system", "rules"), ("human", "facts")])
    assert res.error is None and res.proposal == GOOD
    assert (res.prompt_tokens, res.completion_tokens) == (321, 45)
    req = seen[0]
    assert req["model"] == "granite4:micro" and isinstance(req["format"], dict)          # schema-constrained decoding
    assert req["options"]["num_ctx"] == 8192 and req["options"]["temperature"] == 0.0
    assert [m["role"] for m in req["messages"]] == ["system", "user"]


def test_invalid_output_is_reported_not_raised(fake_ollama):
    url, _ = fake_ollama('{"action": "OFFER", "offer_id": "MAKE_BELIEVE"}')
    res = OllamaLLM(base_url=url).draft([("system", "rules"), ("human", "facts")])
    assert res.proposal is None and res.error


def test_not_json_at_all_is_reported_not_raised(fake_ollama):
    url, _ = fake_ollama("Sure! Here is an offer for you.")
    res = OllamaLLM(base_url=url).draft([("system", "rules"), ("human", "facts")])
    assert res.proposal is None and res.error
