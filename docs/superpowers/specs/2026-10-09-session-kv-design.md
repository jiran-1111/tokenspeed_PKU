# Session KV Reuse Design

## Context

The course requires a multi-turn session to reuse server-side KV: later turns should prefill only newly appended tokens, and a correctness comparison must show that reuse does not change generated token IDs. The current Agent benchmark rebuilds and sends full message history on every turn. Its observed cache hits come from TokenSpeed's generic prefix cache; they do not establish session-bound KV lifetime. The current `RequestHandler` also aborts requests with a non-empty `session_params.id`, and the scheduler does not handle the existing open/close session messages.

## Goal

Add an opt-in, engine-level text-generation session API that retains a session's KV across turns, accepts only appended token IDs after the first turn, reports per-turn prefill/reuse counts, and frees the retained state when the caller closes the session.

## Proposed design

- Extend the native Python `Engine`/`AsyncLLM` interface with `open_session`, session-aware generation, and `close_session`. Session-aware calls carry a stable ID and only the new token IDs; ordinary `generate` calls remain unchanged.
- Keep session state in the scheduler/runtime, not in the benchmark client. It contains the accumulated token context and the KV page/state references needed to continue decode. The next turn attaches to those retained KV pages and prefills only the new suffix. This path must work with generic prefix caching disabled, so a prefix-cache hit alone cannot satisfy the feature.
- Treat an open session as single-writer: reject a second concurrent append for the same ID. Distinct sessions remain independently schedulable. Closing a session releases its retained KV state; an unknown or closed ID returns an explicit error. Shutdown also releases all open session state.
- Keep the external OpenAI/SMG gateway and existing HTTP schema unchanged in this iteration. A dedicated benchmark harness will use the native engine API to isolate session semantics from the locked gateway package. The client may tokenize the full chat-template history to derive the appended suffix, but it sends only that suffix after turn one.
- Add session metadata to native generation results: session ID, total context tokens, newly prefilled tokens, and reused context tokens. Non-session result fields keep their current shape.

## Acceptance criteria

1. With generic prefix caching disabled, a multi-turn session's first turn prefills its full initial context; later turns prefill only the appended suffix and report prior context as reused.
2. A known 4B greedy case produces the same output token IDs through the session path and stateless full-history path.
3. Two interleaved sessions do not share context or KV state; a same-session overlapping append is rejected.
4. Closing a session releases its retained KV allocation, and using that ID afterward is rejected.
5. Unit tests cover lifecycle, suffix accounting, isolation, and errors. A targeted Ascend run records raw per-turn data and compares session-on against stateless full-history output.

## Non-goals

- Changing the external OpenAI/SMG API or modifying locked gateway dependencies.
- Multimodal inputs, multi-writer sessions, automatic idle-session eviction, persistence across process restart, or cross-process session migration.
- Replacing or reimplementing TokenSpeed's scheduler, paged KV allocator, or generic prefix cache.
- Claiming performance benefit before a matched session-on/off benchmark is collected.

## Risks and validation notes

Retaining KV consumes device memory while a session is open; callers must close sessions, and the benchmark must record active session count and HBM/KV usage. Ascend correctness must be gated against the stateless path using a case already known to match the pinned reference; the existing 0.8B and 9B near-tie mismatches cannot serve as the only oracle. The first implementation is limited to the native Python engine API because the HTTP gateway is an external locked dependency.
