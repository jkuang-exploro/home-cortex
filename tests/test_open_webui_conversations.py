"""Execute the shipped greeting hook with a delayed proxy and persisted chat state."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_greeting_creation_waits_for_first_turn_and_is_scoped_to_chat():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js is required to execute the Open WebUI hook')
    patch = Path('docker/cortex/open-webui/home-cortex-greeting.patch').read_text()
    chat = patch.split('+++ b/src/lib/components/chat/Chat.svelte\n', 1)[1]
    source = '\n'.join(line[1:] for line in chat.splitlines() if line.startswith(('+', ' ')))
    hook = source.split('\tconst initializeAgentGreeting =', 1)[1].split('\n\tconst initNewChat', 1)[0]
    submit = source.split('\tconst submitPrompt =', 1)[1].split('\n\t\tconst _files', 1)[0]
    script = r'''
const assert = require('node:assert/strict');
let history = {messages: {}, currentId: null}, params = {};
let selectedModelIds = ['老管家'];
const CORTEX_GREETING_MODEL = '老管家', CORTEX_GREETING_AGENT = 'steward';
const greetingInitializations = new WeakMap();
let greetingInitializedForModel = null;
const $models = [{id: '老管家'}], localStorage = {token: 'verified-session'};
const $i18n = {language: 'en', t: x => x}, toast = {error: () => {}};
let $temporaryChatEnabled = false, saved, creates = 0, pending = [];
const tick = async () => {};
const uuidv4 = () => `message-${creates}`;
const initChatHandler = async () => { saved = JSON.parse(JSON.stringify({history, params})); };
const createCortexConversation = async () => {
  creates++;
  return new Promise(resolve => pending.push(resolve));
};
const reset = () => {
  history = {messages: {}, currentId: null}; params = {};
  greetingInitializedForModel = null;
};
'''
    script += '\nconst initializeAgentGreeting =' + hook
    script += '\nconst submitPrompt =' + submit + '\nreturn params.custom_params?.conversation_id;\n};\n'
    script += r'''
(async () => {
  const greeting = initializeAgentGreeting();
  let sent = false;
  const first = submitPrompt().then(id => { sent = true; return id; });
  await Promise.resolve();
  assert.equal(sent, false);
  assert.equal(creates, 1);
  pending.shift()({id: 'authorized-a', greeting: 'Hello'});
  await greeting;
  assert.equal(await first, 'authorized-a');
  assert.equal(await submitPrompt(), 'authorized-a');
  history = saved.history; params = saved.params; // Open WebUI reloads both.
  assert.equal(await submitPrompt(), 'authorized-a');
  assert.equal(creates, 1);
  reset();
  const second = submitPrompt();
  pending.shift()({id: 'authorized-b', greeting: 'Hello'});
  assert.equal(await second, 'authorized-b');
  assert.equal(creates, 2);
  reset();
  const abandoned = submitPrompt();
  reset(); // Navigate while the proxy is pending.
  const current = submitPrompt();
  pending.shift()({id: 'abandoned', greeting: 'Hello'});
  assert.equal(await abandoned, undefined);
  assert.equal(history.currentId, null);
  pending.shift()({id: 'authorized-c', greeting: 'Hello'});
  assert.equal(await current, 'authorized-c');
  reset();
  const switched = initializeAgentGreeting();
  selectedModelIds = ['another-model'];
  pending.shift()({id: 'wrong-model', greeting: 'Hello'});
  await switched;
  assert.equal(params.custom_params, undefined);
})().catch(error => { console.error(error); process.exitCode = 1; });
'''
    result = subprocess.run([node, '-e', script], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
