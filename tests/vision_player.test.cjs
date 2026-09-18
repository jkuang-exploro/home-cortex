const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const script = fs.readFileSync(path.join(__dirname, '../src/home_cortex/vision/web/vision.js'), 'utf8');

function setup(fetcher = async () => ({ok: true})) {
  const elements = {}, timers = new Map(), requests = [];
  let clock = 0, timerId = 0;
  const element = () => ({
    handlers: {}, children: [], naturalWidth: 0, disabled: true, value: '',
    addEventListener(name, fn) { this.handlers[name] = fn; },
    replaceChildren(...children) { this.children = children; },
    removeAttribute(name) { delete this[name]; },
  });
  for (const id of ['stream-form', 'vision-source', 'vision-key', 'stream-status', 'stream-player', 'stream-disconnect']) elements[id] = element();
  const window = element();
  vm.runInNewContext(script, {
    document: { getElementById: id => elements[id], createElement: element }, window, AbortController,
    fetch: (url, options) => { requests.push({url, ...options, headers: {...options.headers}}); return fetcher(url, options); },
    Date: { now: () => clock },
    setInterval: fn => { timers.set(++timerId, fn); return timerId; },
    clearInterval: id => timers.delete(id),
  });
  return {
    elements, timers, window, requests,
    async connect(key = '', source = '192.168.68.65') {
      elements['vision-key'].value = key;
      elements['vision-source'].value = source;
      await elements['stream-form'].handlers.submit({ preventDefault() {} });
      return elements['stream-player'].children[0];
    },
    tick(ms) { clock += ms; for (const fn of [...timers.values()]) fn(); },
    status: () => elements['stream-status'].textContent,
  };
}

test('connect uses only same-origin session and media, clears key, detects first frame', async () => {
  const app = setup();
  const image = await app.connect('test-key');
  assert.equal(app.requests[0].url, '/vision/session');
  assert.equal(app.requests[0].headers.Authorization, 'Bearer test-key');
  assert.match(app.requests[0].body, /192\.168\.68\.65/);
  assert.equal(app.elements['vision-key'].value, '');
  assert.equal(image.src, '/vision/stream');
  image.naturalWidth = 640;
  app.tick(250);
  assert.match(app.status(), /Preview connected/);
  assert.equal(app.timers.size, 0);
  app.elements['stream-disconnect'].handlers.click();
  assert.equal(image.src, undefined);
  assert.equal(app.elements['stream-player'].children.length, 0);
});

test('session errors prevent media load and show actionable messages', async () => {
  for (const [status, message] of [[401, /API key/], [503, /camera IP/], [422, /Camera address/]]) {
    const app = setup(async () => ({ok: false, status}));
    assert.equal(await app.connect(), undefined);
    assert.match(app.status(), message);
    assert.equal(app.timers.size, 0);
  }
  const app = setup(async () => { throw new Error('offline'); });
  assert.equal(await app.connect(), undefined);
  assert.match(app.status(), /Cannot reach Home Cortex/);
});

test('upstream failure and first-frame timeout clean up and permit retry', async () => {
  const app = setup();
  const first = await app.connect();
  first.onerror();
  assert.match(app.status(), /Home Cortex could not load/);
  assert.equal(first.src, undefined);
  await app.connect();
  app.tick(15000);
  assert.match(app.status(), /Stream unavailable/);
  assert.equal(app.timers.size, 0);
  assert.ok(await app.connect());
});

test('stale callbacks and an aborted session cannot resurrect playback', async () => {
  const app = setup();
  const first = await app.connect();
  const staleError = first.onerror;
  const second = await app.connect();
  staleError();
  assert.equal(app.elements['stream-player'].children[0], second);
  app.window.handlers.pagehide();
  assert.equal(second.src, undefined);
  let resolve;
  const pending = setup(() => new Promise(done => { resolve = done; }));
  const connecting = pending.connect();
  pending.elements['stream-disconnect'].handlers.click();
  resolve({ok: true});
  await connecting;
  assert.equal(pending.elements['stream-player'].children.length, 0);
  assert.equal(pending.requests[0].signal.aborted, true);
});
