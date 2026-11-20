"""A remote stage lost in the middle of a request (recover_wait_s): the server keeps the request, says it is recovering,
waits for the engine and continues the same answer, against fake engines (no GPU).

    python -m unittest serve.test_stage_recovery -v
"""
from __future__ import annotations

import json
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import RECOVERING, ByteTokenizer, EngineDied, MockEngine, RemoteStageLost, Service, serve

ROOT = Path(__file__).resolve().parents[1]
CTX = 8192
LIST = "".join(f"{i}. item {i}\n" for i in range(1, 61))
ANSWER = "</think>\n\n" + LIST
THOUGHT = "Counting to sixty, one line each, nothing skipped. " * 3
TOOL = ("</think>\n\n<tool_call>\n<function=write>\n<parameter=path>\ndoc.md\n</parameter>\n"
        "<parameter=content>\nhello world\n</parameter>\n</function>\n</tool_call>")
TOOLS = [{"type": "function", "function": {"name": "write", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}}}}]
LOST = "remote stage: receive failed: the peer closed the connection"


class DroppingEngine(MockEngine):
    """Plays its script one token a byte, continuing from where the prompt says the answer got to, and loses its
    remote stage once the answer reaches each count in `drops`.  It is down until restart() brings it back."""

    def __init__(self, text, drops=(), error=None, restart_s=0.0, remote_stage=True):
        super().__init__(ByteTokenizer(), text, max_context=CTX)
        self.drops, self.error, self.restart_s, self.remote_stage = list(drops), error, restart_s, remote_stage
        self.prompts, self.up, self.restarts = [], True, 0

    def alive(self):
        return self.up

    def restart(self):
        time.sleep(self.restart_s)
        self.restarts += 1
        self.up = True

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        if not self.up:
            raise EngineDied("the engine is unavailable; this request was not sent")
        self.prompts.append(list(ids))
        done = len(ids) - len(self.prompts[0])
        yield None
        for i, t in enumerate(self.script[done:done + max_new]):
            if self.drops and done + i == self.drops[0]:
                self.drops.pop(0)
                self.up = False
                raise self.error or RemoteStageLost(LOST)
            if cancel.is_set():
                return
            yield t


class WaitingEngine(DroppingEngine):
    """Stays loaded when it loses its remote stage (an engine that says INFO remote_stage_kept=1), and once `drops` are
    used up every request waits for a worker that does not come back: INFO remote_stage_wait= every 50 ms until the
    request is stopped."""

    def alive(self):
        return True

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        if self.drops:
            yield from super().generate(ids, max_new, sampling, cancel)
            return
        while not cancel.is_set():
            time.sleep(0.05)
            self.stage_wait_at = time.time()
            yield None


class Recovery(unittest.TestCase):
    def start(self, engine, wait=5.0):
        """The test's server (made once) on a new engine."""
        self.engine = engine
        if not hasattr(self, "svc"):
            self.svc = Service(engine, engine.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
            httpd = serve(self.svc, port=0)
            self.addCleanup(httpd.server_close)
            self.addCleanup(httpd.shutdown)
            self.base = f"http://127.0.0.1:{httpd.server_address[1]}"
        self.svc.engine, self.svc.recover_wait_s = engine, wait

    def post(self, path, body):
        """-> (status, headers, body text)."""
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), headers={
            "Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.headers, r.read().decode()
        except urllib.error.HTTPError as e:
            with e:
                return e.code, e.headers, e.read().decode()

    def chat(self, stream=True, **extra):
        return self.post("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "count"}],
                                                  "max_tokens": 2000, "stream": stream, **extra})

    @staticmethod
    def chunks(text):
        return [json.loads(line[6:]) for line in text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"]

    def streamed(self, **extra):
        """A chat stream -> (content, the chunks)."""
        code, _, text = self.chat(**extra)
        self.assertEqual(code, 200, text)
        chunks = self.chunks(text)
        content = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c.get("choices"))
        return content, chunks

    def test_a_streamed_answer_continues_whole(self):
        self.start(DroppingEngine(ANSWER, drops=[100]))
        content, chunks = self.streamed()
        self.assertEqual(content, LIST)
        self.assertFalse([c for c in chunks if "error" in c])
        status = [i for i, c in enumerate(chunks) if "strata_status" in c]
        self.assertTrue(status)
        self.assertEqual(chunks[status[0]]["strata_status"]["state"], "recovering")
        self.assertEqual(chunks[status[0]]["strata_status"]["message"], RECOVERING)
        self.assertEqual(chunks[status[0]]["strata_status"]["reason"], LOST)
        self.assertEqual(chunks[status[0]]["choices"][0]["delta"], {})
        said = [i for i, c in enumerate(chunks) if c.get("choices") and c["choices"][0]["delta"].get("content")]
        self.assertLess(said[0], status[0])
        self.assertGreater(said[-1], status[-1])
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(self.engine.restarts, 1)
        self.assertEqual(self.engine.prompts[1], self.engine.prompts[0] + self.engine.script[:100])

    def test_it_dies_twice_and_while_reading_the_prompt(self):
        for drops in ([40, 300], [0], [0, 200]):
            with self.subTest(drops=drops):
                self.start(DroppingEngine(ANSWER, drops=drops))
                content, chunks = self.streamed()
                self.assertEqual(content, LIST)
                self.assertEqual(self.engine.restarts, len(drops))
                self.assertGreaterEqual(len([c for c in chunks if "strata_status" in c]), len(drops))

    def test_not_streamed(self):
        self.start(DroppingEngine(ANSWER, drops=[30, 500]))
        code, _, text = self.chat(stream=False)
        self.assertEqual(code, 200, text)
        self.assertEqual(json.loads(text)["choices"][0]["message"]["content"], LIST)
        self.start(DroppingEngine(ANSWER, drops=[30]))
        code, _, text = self.post("/v1/messages", {"model": "m", "max_tokens": 2000,
                                                   "messages": [{"role": "user", "content": "count"}]})
        self.assertEqual(code, 200, text)
        self.assertEqual("".join(b.get("text", "") for b in json.loads(text)["content"]), LIST)

    def test_messages_and_responses_streams(self):
        self.start(DroppingEngine(ANSWER, drops=[100]))
        code, _, text = self.post("/v1/messages", {"model": "m", "max_tokens": 2000, "stream": True,
                                                   "messages": [{"role": "user", "content": "count"}]})
        self.assertEqual(code, 200, text)
        events = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]
        pings = [e for e in events if e["type"] == "ping" and "strata_status" in e]
        self.assertTrue(pings)
        self.assertEqual(pings[0]["strata_status"]["message"], RECOVERING)
        self.assertEqual("".join(e["delta"].get("text", "") for e in events if e["type"] == "content_block_delta"), LIST)
        self.assertFalse([e for e in events if e["type"] == "error"])
        self.start(DroppingEngine(ANSWER, drops=[100]))
        code, _, text = self.post("/v1/responses", {"model": "m", "max_output_tokens": 2000, "stream": True,
                                                    "input": "count"})
        self.assertEqual(code, 200, text)
        events = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]
        status = [e for e in events if e["type"] == "response.in_progress" and "strata_status" in e]
        self.assertTrue(status)
        self.assertEqual("".join(e["delta"] for e in events if e["type"] == "response.output_text.delta"), LIST)
        self.assertEqual(events[-1]["type"], "response.completed")

    def test_thinking_tool_calls_and_stop_strings_come_out_as_without_a_drop(self):
        cases = [(THOUGHT + ANSWER, {}), (TOOL, {"tools": TOOLS}), ("</think>\n\nabc STOP def", {"stop": ["STOP"]})]
        for script, extra in cases:
            self.start(DroppingEngine(script))
            code, _, want = self.chat(stream=False, **extra)
            self.assertEqual(code, 200, want)
            want = json.loads(want)["choices"]
            for at in range(0, len(script) - 1, 11):
                with self.subTest(script=script[:20], at=at):
                    self.start(DroppingEngine(script, drops=[at]))
                    code, _, got = self.chat(stream=False, **extra)
                    self.assertEqual(code, 200, got)
                    got = json.loads(got)["choices"]
                    for c in want + got:
                        for call in c["message"].get("tool_calls") or []:
                            call.pop("id", None)
                    self.assertEqual(got, want)

    def test_an_engine_that_ended_recovers_only_with_a_remote_stage(self):
        died = EngineDied("the engine stopped unexpectedly (exit code -9)")
        self.start(DroppingEngine(ANSWER, drops=[100], error=died))
        self.assertEqual(self.streamed()[0], LIST)
        self.start(DroppingEngine(ANSWER, drops=[100], error=died, remote_stage=False))
        content, chunks = self.streamed()
        self.assertEqual(content, LIST[:90].rstrip("\n"))
        self.assertIn("exit code -9", chunks[-1]["error"]["message"])
        self.assertEqual(self.engine.restarts, 0)

    def test_off_gives_todays_error(self):
        self.start(DroppingEngine(ANSWER, drops=[100]), wait=0)
        content, chunks = self.streamed()
        self.assertEqual(chunks[-1]["error"]["message"], LOST)
        self.assertFalse([c for c in chunks if "strata_status" in c])
        self.assertEqual(self.engine.restarts, 0)
        self.start(DroppingEngine(ANSWER, drops=[100]), wait=0)
        code, headers, text = self.chat(stream=False)
        self.assertEqual((code, headers["Retry-After"]), (503, "30"))
        self.assertEqual(json.loads(text)["error"]["message"], LOST)

    def test_past_the_limit_gives_todays_error_and_the_turn_waits_for_the_start(self):
        self.start(DroppingEngine(ANSWER, drops=[100], restart_s=1.5), wait=0.3)
        content, chunks = self.streamed()
        self.assertEqual(content, LIST[:90].rstrip("\n"))
        self.assertTrue([c for c in chunks if "strata_status" in c])
        self.assertEqual(chunks[-1]["error"]["message"], LOST)
        self.assertTrue(self.svc.fifo.locked())
        deadline = time.monotonic() + 5
        while self.svc.fifo.locked() and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(self.svc.fifo.locked())
        self.assertEqual(self.engine.restarts, 1)
        self.start(DroppingEngine(ANSWER, drops=[100], restart_s=1.0), wait=0.3)
        code, headers, text = self.chat(stream=False)
        self.assertEqual((code, headers["Retry-After"]), (503, "30"))

    def test_a_second_drop_with_no_new_token_gives_todays_error(self):
        self.start(DroppingEngine(ANSWER, drops=[100, 100]))
        content, chunks = self.streamed()
        self.assertEqual(chunks[-1]["error"]["message"], LOST)
        self.assertEqual(self.engine.restarts, 1)

    def test_a_kept_engine_that_waits_for_its_worker_is_stopped_at_the_limit(self):
        """The engine stays up after the drop, and the continued request waits for a worker that never comes back: it
        ends with the drop's error recover_wait_s after the drop, not when the engine's own wait runs out."""
        self.start(WaitingEngine(ANSWER, drops=[100]), wait=1.0)
        t0 = time.monotonic()
        content, chunks = self.streamed()
        self.assertGreaterEqual(time.monotonic() - t0, 1.0)
        self.assertLess(time.monotonic() - t0, 3.0)
        self.assertEqual(content, LIST[:90].rstrip("\n"))
        self.assertTrue([c for c in chunks if "strata_status" in c])
        self.assertEqual(chunks[-1]["error"]["message"], LOST)
        self.assertEqual(self.engine.restarts, 0)

    def test_a_link_found_lost_at_the_start_counts_from_the_engines_wait(self):
        """The engine's first wait line comes STAGE_WAIT_BEAT_S (10 s) into its wait, so with recover_wait_s 5 the
        request ends at that first line."""
        self.start(WaitingEngine(ANSWER), wait=5.0)
        t0 = time.monotonic()
        code, _, text = self.chat(stream=False)
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertEqual(code, 503)
        self.assertIn("recover_wait_s (5 s)", json.loads(text)["error"]["message"])


if __name__ == "__main__":
    unittest.main()
