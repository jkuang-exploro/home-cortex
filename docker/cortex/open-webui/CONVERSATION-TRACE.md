# Conversation ID propagation audit

Verified against Open WebUI commit `3660bc00fd807deced3400a63bfa6db47811a3bb`
(v0.9.5), with the repository's greeting patch. These are synthetic request
traces and deterministic mocked-interpreter counts, not new GPU measurements.

## Existing normal path

1. A blank steward chat calls the authenticated WebUI proxy:
   `POST /api/v1/cortex/conversations/steward {"language":"en"}`.
2. The proxy supplies the API key and verified user ID/email to
   `POST /agent/steward/conversations`. Cortex resolves the speaker and returns
   `201 {"id":"A", ...}`. No interpreter is needed for creation.
3. The hook stores `A` in `params.custom_params.conversation_id` and stores the
   greeting in history. Upstream `initChatHandler` persists both history and params.
4. Each completion includes `params: {custom_params: {conversation_id: "A"}}`.
   Upstream `utils/middleware.py:apply_params_to_form_data` expands custom params
   into the top-level OpenAI-compatible body. WebUI's own `chat_id` is a separate
   identifier and is not used as a Cortex authorization token.
5. Upstream `loadChat` restores `chatContent.params`. New-chat initialization
   resets params/history and the greeting marker, then creates a separate ID.
6. Cortex checks agent/person ownership before forwarding the ID to the existing
   semantic conversation service. Discourse is additionally keyed by household
   and agent; bound entities are reloaded from authoritative storage.

This normal propagation already worked. No new API or state store was needed.

## Confirmed gap and fix

Before, with a slow creation response:

```text
browser -> proxy: POST conversations/steward                 (pending)
browser: submitPrompt adds the first user message
browser -> completions: params={}                           (ID omitted)
proxy -> browser: 201 id=A
browser: history.currentId exists; discard greeting AND A
next follow-up -> completions: no conversation_id           (2 interpreter calls)
```

The hook now shares one pending initialization per history object. Submission
awaits it before adding the user turn. A response for an abandoned history or a
changed model is discarded before changing current chat state.

```text
browser -> proxy: POST conversations/steward                 (pending)
browser: submitPrompt awaits the existing initialization
proxy -> browser: 201 id=A
browser: save greeting and params.custom_params.conversation_id=A
browser -> completions: conversation_id=A, user="How old is son1?"  (1 call)
reload: restore saved history + params; no new conversation
browser -> completions: conversation_id=A, user="When is his tenth birthday?"
                                                            (1 call)
new chat -> proxy: POST conversations/steward -> 201 id=B
browser -> completions: conversation_id=B, pronoun-only turn  (clarification)
other owner -> completions: conversation_id=A                (404; 0 calls)
```

The API integration tests also exercise `/v1/chat`, `/agent/steward/chat`, and
streaming `/v1/chat/completions`, with the real planner/resolver/executor and a
mocked interpreter transport. The JavaScript hook test executes the shipped
patch with a delayed proxy and a simulated saved-chat reload.

## Loss and adoption boundaries

- Discourse eviction or loss with an authorized ID requires clarification;
  client history does not reconstruct trusted focus. Existing isolation tests
  cover conversation, speaker, household and agent scopes.
- A full API restart or conversation-registry eviction loses authorization too:
  an old ID returns `404 conversation_not_found`, before interpretation. Clients
  must explicitly create a fresh conversation and restate the referent. There
  is no automatic stateless retry or ownership bypass. The WebUI integration
  currently surfaces this error; starting a new chat creates a fresh ID.
- Unpatched WebUI, existing chats without the parameter, switching to the steward
  in an already populated chat, and greeting-proxy failures remain adoption
  boundaries. Use the patched blank-new-chat path, or explicitly create a
  conversation and retain its returned ID in a custom client. Do not substitute
  a client-generated WebUI chat ID or guess a speaker.
- Stateless requests remain supported. A referenced historical antecedent costs
  an extra interpretation; latest-message-only pronouns require clarification.
- Temporary WebUI chats retain params while open but are not saved across reload.

Validation: `459 passed` using `.venv/bin/python -m pytest -q`; the patch applies
cleanly to the pinned upstream source. No live browser deployment, full frontend
production build, or production GPU benchmark was run. The audit's approximately
3.59 s versus 1.76 s remains the prior measured evidence, not a new speedup claim.
