"""A stand-in OpenAI-compatible server for offline tests of src/public/walmart_amazon_dgx.py (scripts/selftest-public-dgx.sh).

FAKE_MODEL      served name
FAKE_MODE       ok | truncate-small | always-truncate | reject-union-schema | garbage-unless-schema
Verdict: the first candidate with confidence 0.9 when its title shares a word with the
query title, else null.  Synthetic -- no model, no reference labels.
"""
import json, os, re, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = os.environ.get("FAKE_MODEL", "gemma4:31b-bf16-vllm")
MODE = os.environ.get("FAKE_MODE", "ok")


def verdict(prompt):
    q = json.loads(re.search(r"Query:\n(.*?)\n\nCandidates:", prompt, re.S).group(1))
    cands = json.loads(prompt.rsplit("Candidates:\n", 1)[1])
    qw = set(re.findall(r"\w+", q["title"].lower()))
    for c in cands:
        if qw & set(re.findall(r"\w+", c["title"].lower())):
            return {"row_id": c["row_id"], "match_confidence": 0.9, "evidence": "shared title words"}
    return {"row_id": None, "match_confidence": None, "evidence": "no overlap"}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        self._send({"data": [{"id": MODEL}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][0]["content"]
        if MODE == "reject-union-schema" and "response_format" in body and \
                "anyOf" not in body["response_format"]["json_schema"]["schema"]:
            return self._send({"error": "grammar compile failed"}, 400)
        content, finish = json.dumps(verdict(prompt)), "stop"
        if MODE == "truncate-small" and body["max_tokens"] < 8192:
            content, finish = content[:20], "length"
        if MODE == "always-truncate":
            content, finish = content[:20], "length"
        if MODE == "garbage-unless-schema" and "response_format" not in body:
            content = "row" + content
        self._send({"model": body["model"], "choices": [{"message": {"content": content},
                    "finish_reason": finish}], "usage": {"prompt_tokens": len(prompt) // 4,
                    "completion_tokens": 40}})


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
