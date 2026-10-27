"""Generate the frozen protocol, prompts, Unicode manifest and schedules. Refuses to overwrite."""
from frozen import (ROOT, TITLE, AUTHOR, FROZEN_TAG, ENDPOINT, ANTHROPIC_VERSION, MODELS, PINNED, EXPERIMENTS, SEEDS,
                    REPLICATES, EXP1_INPUTS, EXP1_MAX_TOKENS, EXP2_SYSTEM, EXP2_INPUTS, EXP2_MAX_TOKENS, EXP3_CELLS,
                    EXP3_STRINGS, EXP3_MAX_TOKENS, EXACT_TARGETS, PREFLIGHT_SYSTEM, PREFLIGHT_INPUT, conditions,
                    build_schedule, schedule_path, digest, pretty, preflight_body)
import classifier
import runner


def prompts():
    return {
        "exp1": {"system": None, "inputs": EXP1_INPUTS},
        "exp2": {"system": EXP2_SYSTEM, "inputs": EXP2_INPUTS},
        "exp3": {"strings": EXP3_STRINGS, "cells": {c: {"system": r, "input": u} for c, (r, u) in EXP3_CELLS.items()}},
        "exact_targets": EXACT_TARGETS,
        "preflight": {"system": PREFLIGHT_SYSTEM, "input": PREFLIGHT_INPUT},
    }


def unicode_manifest():
    strings = {}
    for i, c in enumerate(EXP1_INPUTS, 1):
        strings[f"exp1.input.{i:02d}"] = c
    strings["exp2.system"] = EXP2_SYSTEM
    for i, p in enumerate(EXP2_INPUTS, 1):
        strings[f"exp2.input.{i}"] = p
    for k, v in EXP3_STRINGS.items():
        strings["exp3." + k] = v
    for k, v in EXACT_TARGETS.items():
        strings["target." + k] = v
    strings["preflight.system"] = PREFLIGHT_SYSTEM
    strings["preflight.input"] = PREFLIGHT_INPUT
    return {k: {"literal": v, "codepoints": [f"U+{ord(c):04X}" for c in v], "utf8_hex": v.encode().hex(),
                "byte_length": len(v.encode()), "sha256": digest(v.encode())} for k, v in strings.items()}


def experiments():
    return {
        "1": {"system": None, "inputs": "prompts.json exp1.inputs", "max_tokens": EXP1_MAX_TOKENS, "temperature": 0},
        "2": {"system": "prompts.json exp2.system", "inputs": "prompts.json exp2.inputs", "max_tokens": EXP2_MAX_TOKENS,
              "temperature": 0},
        "3": {"system": "prompts.json exp3.cells.<cell>.system", "inputs": "prompts.json exp3.cells.<cell>.input",
              "cells": {c: [r, u] for c, (r, u) in EXP3_CELLS.items()}, "max_tokens": EXP3_MAX_TOKENS,
              "temperature": "omitted"},
    }


def protocol():
    exps = experiments()
    for e in EXPERIMENTS:
        exps[str(e)].update(seed=SEEDS[e], replicates_per_condition=REPLICATES[e], conditions=len(conditions(e)),
                            requests=len(conditions(e)) * REPLICATES[e], schedule=f"schedules/exp{e}.json",
                            schedule_sha256=digest(schedule_path(e).read_bytes()), classes=classifier.CLASSES[e],
                            classification_priority=classifier.PRIORITY[e])
    return {
        "title": TITLE,
        "author": AUTHOR,
        "frozen_tag": FROZEN_TAG,
        "endpoint": ENDPOINT,
        "request_headers": {"x-api-key": "<credential>", "anthropic-version": ANTHROPIC_VERSION,
                            "content-type": "application/json"},
        "models": MODELS,
        "omitted_parameters": ["stream", "thinking", "top_p", "top_k", "stop_sequences", "tools", "tool_choice",
                               "metadata", "output_config"],
        "request_serialization": "UTF-8 JSON, ensure_ascii=False, sort_keys=True, separators comma/colon",
        "fresh_context": "exactly one user message per request; system only where specified",
        "concurrency": 1,
        "schedule_algorithm": "rows = conditions x replicates in frozen order; Python random.Random(seed).shuffle(rows); saved schedule is authoritative",
        "experiments": exps,
        "requests_total": sum(exps[str(e)]["requests"] for e in EXPERIMENTS),
        "preflight": {"input": PREFLIGHT_INPUT, "system": PREFLIGHT_SYSTEM,
                      "shapes": {str(s): {k: v for k, v in preflight_body(s, "<model>").items() if k not in ("model", "messages")}
                                 for s in EXPERIMENTS},
                      "requests": len(EXPERIMENTS) * len(MODELS),
                      "pass": "HTTP 200, well-formed message, returned model equals requested model, response id begins msg_, request-id header present",
                      "required_before_run": "frozen preflight PASS on the same commit"},
        "retry": {"max_attempts": runner.MAX_ATTEMPTS, "backoff_seconds": runner.BACKOFF_SECONDS,
                  "eligible": ["network timeout", "connection failure", "HTTP 429 not QE", "HTTP 5xx"],
                  "completed_responses": "never retried"},
        "timeouts": {"connect": 30, "read": 600, "write": 30, "pool": 30},
        "stop_rules": {"returned_model_differs": "INCONCLUSIVE", "no_definitive_completion": "INCONCLUSIVE",
                       "frozen_integrity_failure": "INCONCLUSIVE", "existing_output": "refuse"},
        "persisted_response_headers": sorted(runner.RESPONSE_HEADERS),
        "taxonomy": {
            "void": classifier.VOID_TAXONOMY,
            "void_status": classifier.VOID_STATUS,
            "non_void": classifier.NON_VOID_TAXONOMY,
            "cross_script": classifier.CROSS_SCRIPT_TAXONOMY,
            "void_classes": classifier.VOID_CLASSES,
            "visible_text": "concatenation of all text blocks in response order; absent text blocks yield an absent text container",
            "normal_stops": classifier.NORMAL_STOPS,
            "budget_stops": classifier.BUDGET_STOPS,
            "invisible_codepoints": [f"U+{c:04X}" for c in classifier.INVISIBLE_CODEPOINTS],
            "ellipsis_forms": classifier.ELLIPSIS_FORMS,
            "http_errors": {"BE": {"statuses": [400, 413, 422], "markers": classifier.BUDGET_MARKERS},
                            "QE": {"statuses": [400, 402, 429], "markers": classifier.QUOTA_MARKERS},
                            "PI": {"statuses": [400, 404, 405, 413, 422]},
                            "E": "all other non-2xx statuses and transport failures"},
            "exp3_descriptors": "contains (substring presence of each exact target), mixed_script_words (letter-and-mark runs containing both Script=Arabic and Script=Hebrew)",
            "normalization": "none; whitespace and invisible code points are nonempty",
        },
        "dependencies": PINNED,
        "event_hash": "SHA256 of canonical UTF-8 JSON excluding event_hash; first prev_hash 64 zeroes",
    }


def write(name, data):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(data)


def main():
    write("prompts.json", pretty(prompts()))
    write("unicode_manifest.json", pretty(unicode_manifest()))
    for e in EXPERIMENTS:
        write(f"schedules/exp{e}.json", pretty(build_schedule(e)))
    write("protocol.json", pretty(protocol()))


if __name__ == "__main__":
    main()
