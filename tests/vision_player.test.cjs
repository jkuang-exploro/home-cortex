const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const script = fs.readFileSync(path.join(__dirname, '../src/home_cortex/vision/web/vision.js'), 'utf8');

function setup(protocol = 'http:') {
  const elements = {};
  const timers = new Map();
  let clock = 0;
  let timerId = 0;
  const element = () => ({
    handlers: {}, children: [], naturalWidth: 0, disabled: true,
    addEventListener(name, fn) { this.handlers[name] = fn; },
    replaceChildren(...children) { this.children = children; },
    removeAttribute(name) { delete this[name]; },
  });
  for (const id of ['stream-form', 'stream-url', 'stream-status', 'stream-player', 'stream-disconnect']) {
    elements[id] = element();
  }
  const window = element();
  vm.runInNewContext(script, {
    document: { getElementById: id => elements[id], createElement: element },
    location: { protocol }, window, URL,
    Date: { now: () => clock },
    setInterval: fn => { timers.set(++timerId, fn); return timerId; },
    clearInterval: id => timers.delete(id),
  });
  return {
    elements, timers, window,
    connect(url = 'http://127.0.0.1:8088/live.mjpg') {
      elements['stream-url'].value = url;
      elements['stream-form'].handlers.submit({ preventDefault() {} });
      return elements['stream-player'].children[0];
    },
    tick(ms) { clock += ms; for (const fn of [...timers.values()]) fn(); },
    status: () => elements['stream-status'].textContent,
  };
}

test('explicit connect detects first MJPEG frame without a load event; disconnect releases it', () => {
  const app = setup();
  assert.equal(app.elements['stream-player'].children.length, 0);
  const image = app.connect();
  assert.equal(image.src, 'http://127.0.0.1:8088/live.mjpg');
  assert.match(app.status(), /Connecting/);
  image.naturalWidth = 640;
  app.tick(250);
  assert.match(app.status(), /Preview connected/);
  assert.equal(app.timers.size, 0);
  app.elements['stream-disconnect'].handlers.click();
  assert.equal(image.src, undefined);
  assert.equal(app.elements['stream-player'].children.length, 0);
  assert.equal(app.elements['stream-disconnect'].disabled, true);
});

test('errors and first-frame timeout clean up and allow retry', () => {
  const app = setup();
  const first = app.connect();
  first.onerror();
  assert.match(app.status(), /Stream unavailable/);
  assert.equal(first.src, undefined);
  app.connect();
  app.tick(15000);
  assert.match(app.status(), /Stream unavailable/);
  assert.equal(app.timers.size, 0);
  assert.ok(app.connect());
});

test('reconnect ignores stale image callbacks and page exit releases playback', () => {
  const app = setup();
  const first = app.connect();
  const staleError = first.onerror;
  const second = app.connect('http://camera.local:8088/live.mjpg');
  staleError();
  assert.equal(app.elements['stream-player'].children[0], second);
  assert.equal(first.src, undefined);
  app.window.handlers.pagehide();
  assert.equal(second.src, undefined);
  assert.equal(app.timers.size, 0);
});

test('invalid protocols, credential URLs, and mixed content never start playback', () => {
  for (const url of ['not-a-url', 'javascript:alert(1)', 'file:///tmp/image', 'http://user:secret@camera/live.mjpg']) {
    const app = setup();
    assert.equal(app.connect(url), undefined);
    assert.match(app.status(), /without embedded credentials/);
    assert.equal(app.timers.size, 0);
  }
  const app = setup('https:');
  assert.equal(app.connect(), undefined);
  assert.match(app.status(), /HTTPS stream/);
  assert.ok(app.connect('https://camera.example/live.mjpg'));
});
