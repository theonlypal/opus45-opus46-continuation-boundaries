import json
import sys
import tempfile
import unittest
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import runner
import verify
from frozen import ENDPOINT, ANTHROPIC_VERSION, load_schedule


def responder(fail_first=0):
    state = {"calls": 0}

    def handle(request):
        state["calls"] += 1
        assert str(request.url) == ENDPOINT
        assert request.headers["anthropic-version"] == ANTHROPIC_VERSION
        assert request.headers["x-api-key"] == "sk-ant-synthetic"
        if state["calls"] <= fail_first:
            return httpx.Response(529, json={"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})
        body = json.loads(request.content)
        content = [] if "silence" in body["messages"][0]["content"] else [{"type": "text", "text": "x"}]
        return httpx.Response(200, headers={"request-id": "req_synthetic"}, json={
            "id": f"msg_synthetic{state['calls']}", "type": "message", "role": "assistant", "model": body["model"],
            "content": content, "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1}})
    return handle


class OfflineRun(unittest.TestCase):
    def setUp(self):
        runner.BACKOFF_SECONDS = [0, 0, 0]

    def test_full_experiment_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            client = runner.make_client("sk-ant-synthetic", httpx.MockTransport(responder()))
            self.assertTrue(runner.execute(2, out, client, frozen=False))
            status, seen, total, counts = verify.verify_experiment(out, 2)
            self.assertEqual((status, seen, total), ("COMPLETE", 1200, 1200))
            for key, c in counts.items():
                self.assertEqual(dict(c), {"V2": 30} if "silence" in key[1] else {"R": 30})
            self.assertFalse(any(b"sk-ant-synthetic" in p.read_bytes() for p in out.rglob("*") if p.is_file()))
            with self.assertRaises(AssertionError):
                runner.execute(2, out, client, frozen=False)

    def test_retry_then_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            client = runner.make_client("sk-ant-synthetic", httpx.MockTransport(responder(fail_first=2)))
            self.assertTrue(runner.execute(1, out, client, frozen=False))
            first = load_schedule(1)[0]["slot_id"]
            self.assertEqual(len(list((out / "exp1" / "attempts" / first).iterdir())), 3)
            self.assertEqual(verify.verify_experiment(out, 1)[:3], ("COMPLETE", 1080, 1080))

    def test_model_mismatch_is_inconclusive(self):
        def handle(request):
            return httpx.Response(200, json={"id": "msg_x", "type": "message", "role": "assistant", "model": "other",
                                             "content": [], "stop_reason": "end_turn", "usage": {}})
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            client = runner.make_client("sk-ant-synthetic", httpx.MockTransport(handle))
            self.assertFalse(runner.execute(3, out, client, frozen=False))
            completion = json.loads((out / "exp3" / "completion.json").read_bytes())
            self.assertEqual((completion["status"], completion["completed_slots"]), ("INCONCLUSIVE", 1))

    def test_tampering_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            client = runner.make_client("sk-ant-synthetic", httpx.MockTransport(responder()))
            runner.execute(1, out, client, frozen=False)
            slot = load_schedule(1)[5]["slot_id"]
            target = out / "exp1" / "attempts" / slot / "attempt-01" / "response.bin"
            target.chmod(0o644)
            target.write_bytes(target.read_bytes().replace(b'"x"', b'"y"'))
            with self.assertRaises(AssertionError):
                verify.verify_experiment(out, 1)


if __name__ == "__main__":
    unittest.main()
