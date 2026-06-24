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
    "你上一个工具调用缺少 antml: 命名空间前缀，正确写法是 <invoke name=\"...\"> 配 <parameter name=\"...\">。请用正确格式重发。",
    "你上一个工具调用可能过长导致解析失败。请把它拆成多个更小的调用，一次只做一件事，然后重发。",
    "请这一轮只发出一个工具调用（不要并发多个），确保它格式完整、带 antml: 前缀、正确闭合。",
    "重发前请逐一检查每个 <invoke> 和 <parameter> 标签是否都带 antml: 前缀、是否正确闭合。",
    "你把工具调用当成普通文本输出了，它并没有被执行。请改用真正的工具调用语法重新发起。",
    "如果某个工具的调用反复失败，试试换一个等价工具（用 Read 代替 cat、用 Grep 代替 grep）完成同样目的。",
    "先用一句纯文本说明你下一步要调用什么工具、关键参数是什么，然后再发出格式正确的调用。",
    "停一下，放慢节奏。这一轮只需要发出一个最简单的工具调用，先确认它能被正确解析。",
    "Your tool call is missing the required `antml:` namespace prefix. Re-emit it as `<invoke name=\"...\">` with `<parameter name=\"...\">` children.",
    "重发时尽量缩短参数内容（精简命令、少读几行），先让调用成功执行，再逐步补细节。",
    "确认你的工具调用以 </invoke> 正确结尾、每个 parameter 都正确闭合，然后重发。",
    "不要再输出解释性文字，直接发出一个格式完全正确、带 antml: 前缀的工具调用。",
    "把上一个失败的调用整个丢掉，从头重新构造一个干净的工具调用，逐字确认 antml: 前缀没漏。",
    "把当前任务拆成最小的下一步，只为这一步发一个工具调用，成功后再继续。",
    "你的 invoke/parameter 标签很可能漏了 antml: 这几个字符。补上它再发一次。",
    "如果调用里含复杂的 shell 引号或 heredoc，改写成更简单的形式、或先写入临时文件再执行，避免解析歧义。",
    "你上一步没有任何工具被实际执行，任务停在原地。请重新发出该调用以继续。",
    "先发一个你最有把握、最短的工具调用，验证链路通畅，再继续后面的步骤。",
    "这一轮的唯一目标是产出一个语法正确的工具调用——内容可以最简，但格式必须对（带 antml: 前缀）。",
    "不要在普通文字里书写 <invoke> 这类标签；只在真正发起调用时使用它们，并带上 antml: 前缀。",
    "用正确格式重新发起刚才失败的调用，然后继续完成任务，注意不要重复同样的错误。",
    "你的调用可能和前面的文字粘连导致解析失败。让工具调用单独成块、带 antml: 前缀重发。",
    "如果你想一次做很多事，改成分批：先发第一个调用，等结果回来再发下一个。",
    "深呼吸，先想清楚上一个调用为什么没被解析（多半是缺 antml: 前缀或过长），修正后发出一个干净的调用。",
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
                    "畸形 tool call 自动恢复已连续尝试 %d 次仍失败，暂停自动重试、交回人工。"
                    "可手动输入「继续」，或让我把上一个过长的工具调用拆小后重发。" % MAX_RETRIES
                )
            }, ensure_ascii=False))
        except Exception:
            pass
        sys.exit(0)

    prompt, idx = _pick_prompt(st.get("last_idx", -1))
    _write_state(session_id, count + 1, now, idx)
    # Wrap the varied nudge with a stable framing so intent is unambiguous.
    reason = (
        "你上一轮的工具调用没有被正确解析、未执行，任务停在原地。"
        + prompt
        + "（提示 %d/%d）" % (count + 1, MAX_RETRIES)
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
