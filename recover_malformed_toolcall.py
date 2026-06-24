#!/usr/bin/env python3
"""Stop hook: auto-recover from a malformed tool call that was emitted as text.

Failure mode (verified against this machine's real transcripts):
  - The model emits a tool call missing the `antml:` namespace prefix
    (`<invoke name=...>` instead of `<invoke name=...>`). The harness
    cannot parse it, treats it as plain text, retries once internally, and
    when the retry also fails it ENDS the turn.
  - At that point the last assistant message is usually the harness give-up
    text ("...could not be parsed (retry also failed)." /
    "Claude couldn't process that message"), or it is the malformed turn
    itself: a text-only message ending in `</invoke>`.

Strategy: on Stop, read the last assistant message from the transcript. If it
matches the failure signature, return {"decision":"block","reason":...} so the
model re-issues the call correctly and continues. A per-session on-disk counter
(plus a hard cap) prevents an infinite auto-continue loop. ANY error -> exit 0
so a session is never wedged by this hook.
"""
import sys
import os
import json
import re
import time
import random

MAX_RETRIES = 10          # hand back to human only after this many tries in one streak
STREAK_WINDOW_SEC = 600   # gap longer than this => treat next malformed as a fresh streak
COUNTER_DIR = "/tmp"

# A pool of varied recovery nudges. We inject a RANDOM one each time (avoiding an
# immediate repeat) so the model is pushed to break out of a repeating failure
# mode instead of re-reading the same instruction and re-making the same mistake.
# Angles deliberately differ: prefix-fix, split-smaller, switch-tool, think-first,
# one-at-a-time, char-by-char check, minimal-call, avoid-complex-quoting, etc.
RECOVERY_PROMPTS = (
    "Your last tool call is missing the `antml:` namespace prefix. Re-emit it as `<invoke name=\"...\">` with `<parameter name=\"...\">` children, correctly formatted.",
    "Your last tool call may have been too long to parse. Split it into several smaller calls, doing one thing at a time, then resend.",
    "Send exactly ONE tool call this turn (no parallel calls). Make sure it is complete, has the `antml:` prefix, and is properly closed.",
    "Before resending, check each `<invoke>` and `<parameter>` tag individually: does it carry the `antml:` prefix, and is it properly closed?",
    "You emitted the tool call as plain text, so it was never executed. Re-issue it using the real tool-call syntax.",
    "If one tool keeps failing to parse, try an equivalent tool (use Read instead of cat, Grep instead of grep) to accomplish the same thing.",
    "First state in one plain sentence which tool you'll call next and the key arguments, then emit the correctly-formatted call.",
    "Pause and slow down. This turn, just emit the single simplest possible tool call and confirm it parses correctly.",
    "Your tool call is missing the required `antml:` namespace prefix. Re-emit it as `<invoke name=\"...\">` with `<parameter name=\"...\">` children.",
    "When resending, shorten the arguments (trim the command, read fewer lines) so the call executes first; add detail incrementally afterward.",
    "Make sure your tool call ends with a proper `</invoke>` and every parameter is correctly closed, then resend.",
    "Stop emitting explanatory prose — directly emit a single, fully-correct tool call with the `antml:` prefix.",
    "Discard the failed call entirely and construct a fresh, clean tool call from scratch, verifying character-by-character that the `antml:` prefix is present.",
    "Break the task into the smallest possible next step and send a tool call for just that step; continue once it succeeds.",
    "Your invoke/parameter tags are most likely missing the literal `antml:` characters. Add them and send again.",
    "If the call contains complex shell quoting or a heredoc, rewrite it more simply, or write to a temp file first, to avoid parsing ambiguity.",
    "No tool actually executed on your last step, so the task is stuck in place. Re-issue the call to continue.",
    "Send the shortest tool call you're most confident in first to confirm the pipeline works, then continue with the remaining steps.",
    "The only goal this turn is to produce a syntactically valid tool call — the content can be minimal, but the format must be correct (with the `antml:` prefix).",
    "Don't write tags like `<invoke>` in ordinary prose; only use them when actually making a call, and include the `antml:` prefix.",
    "Re-issue the call that just failed, using the correct format, then continue the task — and avoid repeating the same mistake.",
    "Your call may have run together with the preceding text and failed to parse. Put the tool call in its own block, with the `antml:` prefix, and resend.",
    "If you're trying to do many things at once, switch to batches: send the first call, wait for its result, then send the next.",
    "Take a breath and first reason about WHY the last call didn't parse (most likely a missing `antml:` prefix or it was too long), then emit one clean call.",
)

# Signal A: harness give-up / retry-failed markers (very specific -> ~0 false positives)
GIVEUP_MARKERS = (
    "tool call could not be parsed",
    "could not be parsed (retry",
    "tool call was malformed",
    "couldn't process that message",
    "could not process that message",
)

_INVOKE_RE = re.compile(r"<(?:antml:)?invoke\s+name=", re.IGNORECASE)
_PARAM_RE = re.compile(r"<(?:antml:)?parameter\s+name=", re.IGNORECASE)

# Strip markdown code so XML/markers we are merely *discussing* (wrapped in
# backticks, like this very hook's own conversation) don't self-trigger.
# Bare leaked tool-call XML is NOT wrapped in backticks, so it survives.
_FENCED_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_RE = re.compile(r"`[^`]*`")


def _strip_code(text):
    if not text:
        return text
    t = _FENCED_RE.sub(" ", text)
    t = _INLINE_RE.sub(" ", t)
    return t


def _state_path(session_id):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id or "unknown")
    return os.path.join(COUNTER_DIR, "cc-recover-%s.json" % safe)


def _read_state(session_id):
    """Return {count, ts, last_idx}. Missing/corrupt -> fresh zero state."""
    try:
        with open(_state_path(session_id)) as f:
            d = json.load(f)
        return {
            "count": int(d.get("count", 0)),
            "ts": float(d.get("ts", 0.0)),
            "last_idx": int(d.get("last_idx", -1)),
        }
    except Exception:
        return {"count": 0, "ts": 0.0, "last_idx": -1}


def _write_state(session_id, count, ts, last_idx):
    try:
        with open(_state_path(session_id), "w") as f:
            json.dump({"count": count, "ts": ts, "last_idx": last_idx}, f)
    except Exception:
        pass


def _reset_state(session_id):
    try:
        os.remove(_state_path(session_id))
    except Exception:
        pass


def _now():
    try:
        return time.time()
    except Exception:
        return 0.0


def _pick_prompt(last_idx):
    """Random recovery prompt, avoiding an immediate repeat of last_idx.
    Returns (prompt_text, chosen_idx)."""
    n = len(RECOVERY_PROMPTS)
    if n == 1:
        return RECOVERY_PROMPTS[0], 0
    try:
        idx = random.randrange(n)
        if idx == last_idx:
            idx = (idx + 1) % n
    except Exception:
        idx = 0 if last_idx != 0 else 1
    return RECOVERY_PROMPTS[idx], idx


def _heartbeat(session_id, malformed, has_tool_use, text):
    """If the sentinel file /tmp/cc-recover-heartbeat.on exists, append one line
    per Stop invocation so hook activation is verifiable from outside the model.
    Delete the sentinel to disable (zero behavior otherwise). Never raises."""
    try:
        switch = os.path.join(COUNTER_DIR, "cc-recover-heartbeat.on")
        if not os.path.exists(switch):
            return
        from datetime import datetime
        ts = datetime.now().isoformat(timespec="seconds")
        tl = len(text) if text else 0
        line = "%s session=%s malformed=%s tool_use=%s text_len=%d\n" % (
            ts, session_id, malformed, has_tool_use, tl)
        with open(os.path.join(COUNTER_DIR, "cc-recover-heartbeat.log"), "a") as f:
            f.write(line)
    except Exception:
        pass


def _last_assistant(transcript_path):
    """Return (text, has_tool_use_block) for the last assistant turn, or (None, False)."""
    if not transcript_path or not os.path.exists(transcript_path):
        return None, False
    last = None
    try:
        with open(transcript_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("type") == "assistant":
                    last = obj
    except Exception:
        return None, False
    if last is None:
        return None, False
    msg = last.get("message", {}) or {}
    content = msg.get("content", [])
    if isinstance(content, str):
        return content, False
    text_parts = []
    has_tool_use = False
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "tool_use":
            has_tool_use = True
        elif btype == "text":
            text_parts.append(block.get("text", ""))
    return "".join(text_parts), has_tool_use


def _is_malformed_stop(text, has_tool_use):
    # A real malformed-leak turn never carries a parsed tool_use block.
    if not text or has_tool_use:
        return False
    # Strip backticked code first: markers/XML we are merely *discussing*
    # (e.g. this hook's own design chat) are fenced or inline-coded and get
    # removed; genuinely leaked tool-call XML is bare and survives.
    clean = _strip_code(text)
    low = clean.lower()
    # Signal A: explicit harness give-up / malformed markers (in prose, not code).
    if any(m in low for m in GIVEUP_MARKERS):
        return True
    # Signal B: the turn ENDS in a bare tool-call close tag and contains
    # bare invoke/parameter XML -> the malformed call leaked as text.
    # Requiring the close tag to be the very tail (after code-strip) excludes
    # normal answers that merely quote a tag mid-sentence.
    stripped = clean.rstrip()
    if stripped.endswith("</invoke>") or stripped.endswith("</invoke>"):
        if _INVOKE_RE.search(clean) or _PARAM_RE.search(clean):
            return True
    return False


def main():
    try:
        raw = sys.stdin.read()
    except Exception:
        sys.exit(0)
    try:
        data = json.loads(raw) if raw and raw.strip() else {}
    except Exception:
        sys.exit(0)

    session_id = data.get("session_id") or "unknown"
    transcript_path = os.path.expanduser(data.get("transcript_path") or "")

    text, has_tool_use = _last_assistant(transcript_path)
    malformed = _is_malformed_stop(text, has_tool_use)
    _heartbeat(session_id, malformed, has_tool_use, text)

    now = _now()
    st = _read_state(session_id)

    if not malformed:
        # Normal stop -> clear any prior recovery streak and let it stop.
        _reset_state(session_id)
        sys.exit(0)

    # Streak accounting: a malformed stop continues the prior streak only if it
    # happened within STREAK_WINDOW_SEC of the last one. A long human-idle gap
    # (e.g. picking the conversation back up hours later) starts a fresh streak,
    # so the retry budget is per-burst, not per-session-lifetime. Crucially we do
    # NOT reset on an interleaved success -> Opus drops the antml: prefix
    # intermittently (fail, ok, fail, ok), and resetting on each ok would keep
    # the streak near zero forever and the MAX_RETRIES brake would never engage.
    within_window = st["ts"] and (now - st["ts"]) <= STREAK_WINDOW_SEC
    count = st["count"] if within_window else 0

    if count >= MAX_RETRIES:
        # Tried enough in this burst; hand back to the human instead of looping.
        _reset_state(session_id)
        try:
            print(json.dumps({
                "systemMessage": (
                    "Auto-recovery from a malformed tool call has failed %d times "
                    "in a row; pausing automatic retries and handing back to you. "
                    "Type 'continue' to retry, or ask me to split the previous "
                    "(possibly too-long) tool call into smaller ones." % MAX_RETRIES
                )
            }))
        except Exception:
            pass
        sys.exit(0)

    prompt, idx = _pick_prompt(st.get("last_idx", -1))
    _write_state(session_id, count + 1, now, idx)
    # Wrap the varied nudge with a stable framing so intent is unambiguous.
    reason = (
        "Your last tool call was not parsed and did not execute; the task is "
        "stuck in place. "
        + prompt
        + " (recovery attempt %d/%d)" % (count + 1, MAX_RETRIES)
    )
    try:
        print(json.dumps({"decision": "block", "reason": reason}, ensure_ascii=False))
    except Exception:
        sys.exit(0)
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Never wedge a session because of this hook.
        sys.exit(0)
