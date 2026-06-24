# claude-code-toolcall-recovery

A [Claude Code](https://docs.claude.com/en/docs/claude-code) **Stop hook** that automatically recovers from *malformed tool calls* — the failure mode where the model emits a tool call as plain text (most often by dropping the `antml:` namespace prefix), the harness can't parse it, its built-in retry also fails, and **the turn just ends, leaving the session stuck waiting for you to type "continue".**

If you've ever watched Claude Code stop dead mid-task and had to babysit it with repeated "continue" nudges, this hook is for you.

## What it does

On every `Stop` event, the hook reads the last assistant message from the transcript. If it matches the malformed-tool-call signature, it returns a `block` decision with an instruction telling the model to re-issue the call correctly — so the session **continues on its own** instead of stalling.

```
model emits malformed tool call  ─┐
   (e.g. <invoke …> with no antml:)│
harness can't parse → retries once │   without hook: turn ends, you type "continue"
   retry also fails → turn ENDS    │   with hook:    Stop fires → block → model retries → continues
                                   ─┘
```

### Before / after

```
WITHOUT the hook                         WITH the hook
────────────────────                     ────────────────────
⏺ (malformed tool call)                  ⏺ (malformed tool call)
✗ tool call could not be parsed          ✗ tool call could not be parsed
  (retry also failed)                      (retry also failed)
                                         ↻ Stop hook: "re-issue the call
  ⏸  session idle…                          correctly (it's missing the
                                            antml: prefix); continue."
  you: continue                          ⏺ (correct tool call) → runs
  you: continue  …again                  ✓ continues the task on its own
```

> A screen recording is worth more than this sketch — a GIF/asciinema demo is on the to-do list.

### Key design points

- **Two detection signals** (either one triggers recovery):
  - **A** — the last message contains an explicit harness give-up marker (`could not be parsed`, `couldn't process that message`, `malformed`, …).
  - **B** — the last message has *no* parsed tool-use block, yet its text ends in a bare `</invoke>` / `</invoke>` and contains tool-call XML — i.e. the call leaked out as text.
- **False-positive guard** — markdown code (fenced ` ``` ` blocks and inline `` ` `` spans) is stripped *before* matching. This is what lets you safely *discuss* tool-call XML (like this very README, or a conversation about the hook itself) without the hook firing on your own words. Genuinely leaked XML is bare and survives the strip.
- **Varied recovery prompts** — instead of injecting the same instruction every time (which tends to make the model repeat the same mistake), the hook draws from a pool of ~24 differently-angled nudges (fix the prefix, split the call smaller, switch tools, think first, one call at a time, check each tag, …), picking a random one each time and avoiding an immediate repeat. This helps break the model out of a repeating failure loop.
- **Time-windowed retry budget** — a per-session streak counter caps automatic retries (default **10**) before handing back to you. The streak is *time-windowed*: malformed stops within ~10 minutes count as one burst; a long idle gap starts fresh. Critically, the counter is **not** reset on an interleaved success — intermittent failures (fail, ok, fail, ok) would otherwise keep the counter near zero forever and defeat the cap.
- **Fail-safe** — any internal error makes the hook exit 0. It can never wedge a session by itself.

### What it can't do

- It can't recover from genuine **API errors** (rate limits, 5xx). Those surface through a different event the hook has no say over.
- It can't fix a model that keeps failing the *same* way indefinitely — it can only keep nudging (with varied prompts) up to the retry budget, then hand back to you.

## Install

Requires Python 3 (standard library only — no dependencies) and Claude Code.

1. Copy the hook script somewhere stable, e.g.:

   ```bash
   mkdir -p ~/.claude/hooks
   cp recover_malformed_toolcall.py ~/.claude/hooks/
   ```

2. Register it as a `Stop` hook in your Claude Code settings (`~/.claude/settings.json`):

   ```json
   {
     "hooks": {
       "Stop": [
         {
           "hooks": [
             {
               "type": "command",
               "command": "python3 ~/.claude/hooks/recover_malformed_toolcall.py",
               "timeout": 10
             }
           ]
         }
       ]
     }
   }
   ```

   > If you already have other `Stop` hooks, add this one to the existing array rather than replacing it.

3. Restart Claude Code (or open the `/hooks` menu once) so the config watcher reloads. The hook takes effect on **new** sessions.

## Configuration

Edit the constants at the top of `recover_malformed_toolcall.py`:

| Constant | Default | Meaning |
|---|---|---|
| `MAX_RETRIES` | `10` | Auto-retries per burst before handing back to you. |
| `STREAK_WINDOW_SEC` | `600` | Malformed stops within this many seconds count as one burst. |
| `RECOVERY_PROMPTS` | 24 prompts | The pool of recovery nudges. Add/remove freely. |

## Verifying it's actually loaded

Hooks run in the harness, *after* the model's turn — so the model can't "see" whether it's installed. To get external proof, the script has an opt-in heartbeat probe:

```bash
touch /tmp/cc-recover-heartbeat.on      # enable
# … use Claude Code …
cat /tmp/cc-recover-heartbeat.log       # one line per Stop event
rm /tmp/cc-recover-heartbeat.on         # disable (zero overhead otherwise)
```

Each log line records the session id, whether the turn was judged malformed, whether a tool-use block was present, and the text length.

## How it works (internals)

State lives in `/tmp/cc-recover-<session>.json` as `{count, ts, last_idx}`. On each `Stop`:

1. Read the last assistant turn from the transcript.
2. Decide malformed-or-not via the two signals above (after stripping code).
3. If not malformed → clear the streak, let the session stop normally.
4. If malformed → if within the time window and `count >= MAX_RETRIES`, emit a `systemMessage` handing back to you; otherwise pick a random recovery prompt, increment the streak, and emit a `block` decision with that prompt.

## Related issues

This hook is a community workaround for a widely-reported failure mode. If you landed here from one of these, you're in the right place:

- [anthropics/claude-code#49747](https://github.com/anthropics/claude-code/issues/49747) — Opus mixes legacy XML tool-use format into tool calls on longer payloads (the root cause)
- [anthropics/claude-code#62123](https://github.com/anthropics/claude-code/issues/62123) — "Model's tool call could not be parsed (retry also failed)"
- [anthropics/claude-code#63875](https://github.com/anthropics/claude-code/issues/63875) — Recurring "tool call could not be parsed" error
- [anthropics/claude-code#64500](https://github.com/anthropics/claude-code/issues/64500) — Tool call parsing repeatedly fails causing extended session hang

This is an unofficial, third-party tool — not affiliated with or endorsed by Anthropic. It treats the *symptom* (a stalled session) so you don't have to babysit it; it does not fix the underlying parser behavior.

## License

[MIT](LICENSE) © Mufan Qiu
