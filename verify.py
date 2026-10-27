"""Offline verification of a run output. No network operations."""
import argparse
import collections
import json
from pathlib import Path
import sys
from classifier import classify
from frozen import EXPERIMENTS, canonical, digest, request_bytes, load_schedule


def condition_key(experiment, slot):
    if experiment == 1:
        return (slot["model"], slot["input"])
    if experiment == 2:
        return (slot["model"], slot["input"], slot["max_tokens"])
    return (slot["model"], slot["cell"])


def check_chain(path):
    previous, seq, rows = "0" * 64, 0, []
    for line in path.read_bytes().splitlines():
        row = json.loads(line)
        seq += 1
        assert row["seq"] == seq, "Event sequence gap"
        assert row["prev_hash"] == previous, "Event chain break"
        expected = row.pop("event_hash")
        assert digest(canonical(row)) == expected, "Event hash mismatch"
        previous = expected
        rows.append(row)
    return rows, previous


def check_attempt(experiment, folder, body):
    meta = json.loads((folder / "attempt.json").read_bytes())
    request = (folder / "request.bin").read_bytes()
    response = (folder / "response.bin").read_bytes()
    assert request == body, "Request bytes differ from frozen request"
    assert digest(request) == meta["request_sha256"], "Request hash mismatch"
    assert digest(response) == meta["response_sha256"], "Response hash mismatch"
    assert meta["request_headers"].get("x-api-key") in (None, "REDACTED"), "Credential persisted"
    result = classify(experiment, meta["http_status"], response)
    assert result == json.loads((folder / "classification.json").read_bytes()), "Classification does not reproduce"
    return meta, result


def verify_experiment(output, experiment):
    folder = output / f"exp{experiment}"
    schedule = load_schedule(experiment)
    events, head = check_chain(folder / "events.jsonl")
    completion = json.loads((folder / "completion.json").read_bytes())
    assert completion["event_chain_head"] == head, "Completion does not match event chain head"
    completed = {e["data"]["slot_id"]: e["data"] for e in events if e["event"] == "slot_completed"}
    persisted = collections.Counter(e["data"]["slot_id"] for e in events if e["event"] == "attempt_persisted")
    counts = collections.defaultdict(collections.Counter)
    seen = 0
    for slot in schedule:
        attempts_dir = folder / "attempts" / slot["slot_id"]
        if not attempts_dir.exists():
            break
        body = request_bytes(experiment, slot)
        attempts = sorted(attempts_dir.iterdir())
        assert len(attempts) == persisted[slot["slot_id"]], "Attempt folders differ from event log"
        for path in attempts:
            meta, result = check_attempt(experiment, path, body)
        record_path = folder / "definitive" / (slot["slot_id"] + ".json")
        if record_path.exists():
            record = json.loads(record_path.read_bytes())
            assert record == completed[slot["slot_id"]], "Definitive record differs from event log"
            assert record["attempt"] == len(attempts), "Definitive record is not the last attempt"
            assert record["class"] == result["class"] and record["response_sha256"] == meta["response_sha256"]
            assert record["requested_model"] == slot["model"]
            counts[condition_key(experiment, slot)][record["class"]] += 1
            seen += 1
    assert seen == len(completed) == completion["completed_slots"], "Completed slot count mismatch"
    return completion["status"], seen, len(schedule), counts


def verify_preflight(output):
    verdict = json.loads((output / "preflight" / "verdict.json").read_bytes())
    for check in verdict["checks"]:
        folder = output / "preflight" / f"shape{check['shape']}" / check["model"]
        meta = json.loads((folder / "attempt.json").read_bytes())
        assert digest((folder / "request.bin").read_bytes()) == meta["request_sha256"] == check["request_sha256"]
        assert digest((folder / "response.bin").read_bytes()) == meta["response_sha256"]
    return verdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--experiment", type=int, choices=EXPERIMENTS)
    args = p.parse_args()
    output = args.output.resolve()
    failures = 0
    if (output / "preflight" / "verdict.json").exists():
        verdict = verify_preflight(output)
        print(f"preflight {verdict['status']} frozen={verdict['frozen']} commit={verdict['commit']}")
    for experiment in [args.experiment] if args.experiment else EXPERIMENTS:
        if not (output / f"exp{experiment}").exists():
            continue
        try:
            status, seen, total, counts = verify_experiment(output, experiment)
        except AssertionError as exc:
            failures += 1
            print(f"exp{experiment} VERIFY FAIL: {exc}")
            continue
        print(f"exp{experiment} VERIFY PASS run={status} {seen}/{total}")
        for key in sorted(counts, key=str):
            print("  " + " | ".join(map(str, key)) + " | " + " ".join(f"{c}={n}" for c, n in sorted(counts[key].items())))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
