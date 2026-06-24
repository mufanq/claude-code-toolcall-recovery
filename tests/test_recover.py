#!/usr/bin/env python3
"""Self-contained regression tests for recover_malformed_toolcall.py.

No external fixtures: every transcript is synthesized inline, so this runs
anywhere with just Python 3 (standard library). Run:

    python3 tests/test_recover.py
"""
import os
import sys
import json
import tempfile
import importlib.util

HOOK = os.path.join(os.path.dirname(__file__), "..", "recover_malformed_toolcall.py")
spec = importlib.util.spec_from_file_location("rec", HOOK)
rec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rec)

_failures = []


def check(name, cond):
    mark = "ok  " if cond else "FAIL"
    print("  [%s] %s" % (mark, name))
    if not cond:
        _failures.append(name)


def _assistant(text, tool_use=False):
    content = [{"type": "text", "text": text}]
    if tool_use:
        content.append({"type": "tool_use", "name": "Bash", "input": {}})
    return {"type": "assistant", "message": {"content": content}}


def _write_transcript(rows):
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


# --------------------------------------------------------------------------
def test_detection():
    print("detection: _is_malformed_stop")
    MALFORMED_XML = ('let me check\n<' 'invoke name="Bash">\n'
                     '<' 'parameter name="command">ls</parameter>\n</invoke>')
    check("bare leaked XML ending in </invoke> -> malformed",
          rec._is_malformed_stop(MALFORMED_XML, False) is True)
    check("harness give-up marker -> malformed",
          rec._is_malformed_stop(
              "The model's tool call could not be parsed (retry also failed).",
              False) is True)
    check("turn carrying a real tool_use block -> NOT malformed",
          rec._is_malformed_stop(MALFORMED_XML, True) is False)
    check("plain answer, no XML -> NOT malformed",
          rec._is_malformed_stop("All done. The hook is ready.", False) is False)
    # The crucial false-positive guard: discussing tool-call XML in backticks.
    check("XML wrapped in inline backticks -> NOT malformed",
          rec._is_malformed_stop(
              'use `<' 'invoke name=...>` not `<' 'invoke name=...>`. fixing now.',
              False) is False)
    check("XML inside a fenced code block -> NOT malformed",
          rec._is_malformed_stop(
              'I added a probe:\n```python\n<' 'invoke name="x">\n</invoke>\n```\ndone.',
              False) is False)
    check("give-up phrase quoted inside backticks -> NOT malformed",
          rec._is_malformed_stop(
              'the log line `...could not be parsed (retry also failed).` shows it.',
              False) is False)


# --------------------------------------------------------------------------
def test_prompt_pool():
    print("prompt pool: _pick_prompt")
    check("pool has >= 20 prompts", len(rec.RECOVERY_PROMPTS) >= 20)
    last = -1
    seen = set()
    immediate_repeats = 0
    for _ in range(300):
        prompt, idx = rec._pick_prompt(last)
        if idx == last:
            immediate_repeats += 1
        check_prompt_matches = prompt == rec.RECOVERY_PROMPTS[idx]
        if not check_prompt_matches:
            _failures.append("prompt/idx mismatch")
        seen.add(idx)
        last = idx
    check("never an immediate repeat over 300 draws", immediate_repeats == 0)
    check("random draws cover >= 15 distinct prompts", len(seen) >= 15)


# --------------------------------------------------------------------------
def test_state_roundtrip():
    print("state: read/write/reset")
    sid = "unit-roundtrip"
    rec._reset_state(sid)
    check("missing state reads as count=0", rec._read_state(sid)["count"] == 0)
    rec._write_state(sid, 4, 1234.5, 9)
    st = rec._read_state(sid)
    check("count persisted", st["count"] == 4)
    check("ts persisted", st["ts"] == 1234.5)
    check("last_idx persisted", st["last_idx"] == 9)
    rec._reset_state(sid)
    check("reset clears state", rec._read_state(sid)["count"] == 0)


# --------------------------------------------------------------------------
def _run_hook(session_id, transcript_path):
    """Invoke main() in-process by faking stdin/stdout."""
    import io
    payload = json.dumps({"session_id": session_id,
                          "transcript_path": transcript_path})
    old_in, old_out = sys.stdin, sys.stdout
    sys.stdin = io.StringIO(payload)
    sys.stdout = io.StringIO()
    code = 0
    try:
        try:
            rec.main()
        except SystemExit as e:
            code = e.code or 0
        out = sys.stdout.getvalue()
    finally:
        sys.stdin, sys.stdout = old_in, old_out
    try:
        resp = json.loads(out) if out.strip() else {}
    except Exception:
        resp = {"_raw": out}
    return resp, code


def test_end_to_end_budget():
    print("end-to-end: block N times then hand back")
    MALFORMED_XML = ('check\n<' 'invoke name="Bash">\n'
                     '<' 'parameter name="command">ls</parameter>\n</invoke>')
    tp = _write_transcript([_assistant(MALFORMED_XML, tool_use=False)])
    sid = "e2e-budget"
    rec._reset_state(sid)
    blocks = 0
    giveup_at = None
    reasons = set()
    for n in range(1, rec.MAX_RETRIES + 5):
        resp, _ = _run_hook(sid, tp)
        if "decision" in resp and resp["decision"] == "block":
            blocks += 1
            reasons.add(resp.get("reason", ""))
        elif "systemMessage" in resp:
            giveup_at = n
            break
    check("blocked exactly MAX_RETRIES times", blocks == rec.MAX_RETRIES)
    check("handed back on attempt MAX_RETRIES+1",
          giveup_at == rec.MAX_RETRIES + 1)
    check("varied reasons were injected (>=5 distinct)", len(reasons) >= 5)
    rec._reset_state(sid)
    os.remove(tp)


def test_normal_stop_passes():
    print("end-to-end: a clean turn is left alone")
    tp = _write_transcript([_assistant("All done — everything is wired up.", False)])
    sid = "e2e-clean"
    rec._reset_state(sid)
    resp, code = _run_hook(sid, tp)
    check("no block on a normal stop", "decision" not in resp)
    check("exit code 0", code == 0)
    rec._reset_state(sid)
    os.remove(tp)


def test_streak_window():
    print("streak window: stale streak restarts")
    sid = "window"
    rec._reset_state(sid)
    rec._write_state(sid, rec.MAX_RETRIES, 1.0, 3)  # ancient ts
    now = rec._now()
    st = rec._read_state(sid)
    within = bool(st["ts"]) and (now - st["ts"]) <= rec.STREAK_WINDOW_SEC
    check("ancient streak falls outside the window", within is False)
    rec._reset_state(sid)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    test_detection()
    test_prompt_pool()
    test_state_roundtrip()
    test_end_to_end_budget()
    test_normal_stop_passes()
    test_streak_window()
    print()
    if _failures:
        print("FAILED (%d):" % len(_failures))
        for f in _failures:
            print("  - %s" % f)
        sys.exit(1)
    print("All tests passed.")
