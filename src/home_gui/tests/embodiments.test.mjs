import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { createServer } from 'vite';

let vite;
let render;
let Embodiments;
let Chat;
let navigation;

const noop = () => {};
const base = {
  id: 'embodiment:microduck-01',
  name: 'MicroDuck',
  agent: { id: 'agent:butler', name: '老管家' },
  linked: true,
  connected: false,
  state: 'linked_offline',
  runtime: { connected_at: null, last_seen: null },
  telemetry: {
    available: false, valid: false, fresh: false,
    space_id: null, measured_at: null,
  },
  capabilities: [
    { name: 'mobility.move', supported: true, available: false },
    { name: 'vision.observe', supported: true, available: false },
  ],
};

function body(overrides = {}) {
  return {
    ...base,
    ...overrides,
    runtime: { ...base.runtime, ...overrides.runtime },
    telemetry: { ...base.telemetry, ...overrides.telemetry },
    capabilities: overrides.capabilities ?? base.capabilities,
  };
}

function page(items, selectedId = null, detail = null, detailError = '') {
  return render(Embodiments, { props: {
    language: 'en', items, detail, selectedId, detailError,
    loading: false, error: '', onselect: noop, onback: noop, onrefresh: noop,
  } }).body;
}

function chat(items, activeEmbodimentId = null) {
  return render(Chat, { props: {
    language: 'en', models: [{ id: '老管家', owned_by: 'home-cortex', kind: 'agent' }],
    model: '老管家', messages: [], pending: false,
    embodiments: items, activeEmbodimentId,
    activeAgentId: 'steward', activeAgentEntityId: 'agent:butler',
    onmodel: noop, onlanguage: noop, onsend: noop, onembodiment: noop,
    onrefresh: noop, onopenbody: noop,
  } }).body;
}

before(async () => {
  vite = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
  ({ render } = await vite.ssrLoadModule('svelte/server'));
  ({ default: Embodiments } = await vite.ssrLoadModule('/src/components/Embodiments.svelte'));
  ({ default: Chat } = await vite.ssrLoadModule('/src/components/Chat.svelte'));
  navigation = await vite.ssrLoadModule('/src/lib/navigation.ts');
});

after(async () => { await vite?.close(); });

test('routes preserve embodiment IDs and keep Media and chat paths', () => {
  assert.deepEqual(navigation.routeFromPath('/embodiments'),
    { page: 'embodiments', embodimentId: null });
  const path = navigation.pathForRoute({ page: 'embodiments', embodimentId: base.id });
  assert.deepEqual(navigation.routeFromPath(path),
    { page: 'embodiments', embodimentId: base.id });
  assert.equal(navigation.routeFromPath('/media').page, 'media');
  assert.equal(navigation.routeFromPath('/').page, 'chat');
});

test('list distinguishes unlinked, linked offline, and online without pose', () => {
  const unlinked = body({ id: 'embodiment:unlinked', agent: null, linked: false, state: 'unlinked' });
  const online = body({ connected: true, state: 'linked_online',
    runtime: { last_seen: '2026-09-28T18:00:00Z' } });
  const html = page([unlinked, base, online]);
  assert.match(html, /Not linked to an agent/);
  assert.match(html, /Linked to: 老管家/);
  assert.match(html, /Online/);
  assert.match(html, /Offline/);
  assert.match(html, /Position currently unavailable/);
});

test('detail shows geometry, per-capability state, and p95 values', () => {
  const transform = Object.fromEntries(
    ['x', 'y', 'z', 'yaw', 'pitch', 'roll'].map((name) =>
      [name, { value: name === 'x' ? 1.372 : 0, p95: 0.024 }]),
  );
  const detail = body({
    connected: true, state: 'linked_online',
    runtime: { connected_at: '2026-09-28T18:00:00Z', last_seen: '2026-09-28T18:01:00Z' },
    telemetry: {
      available: true, valid: true, fresh: true,
      space_id: 'space:kitchen', measured_at: '2026-09-28T18:01:00Z',
      estimate: { embodiment_id: base.id, space_id: 'space:kitchen',
        measured_at: '2026-09-28T18:01:00Z', validity: 'valid', transform },
    },
    capabilities: [
      { name: 'mobility.move', supported: true, available: true },
      { name: 'audio.speak', supported: true, available: false },
    ],
    geometry: { box: { length_m: 0.32, width_m: 0.24, height_m: 0.18,
      center: { x: 0.04, y: 0, z: 0.09 } } },
    local_frame: { forward: '+x', left: '+y', up: '+z' },
  });
  const html = page([detail], detail.id, detail);
  assert.match(html, /0\.32 × 0\.24 × 0\.18 m/);
  assert.match(html, /mobility\.move/);
  assert.match(html, /audio\.speak/);
  assert.match(html, /Unavailable/);
  assert.match(html, /1\.372 ± 0\.024 m/);
  assert.match(html, /Kitchen/);
});

test('online no-estimate detail remains online and hides position values', () => {
  const detail = body({
    connected: true, state: 'linked_online',
    telemetry: { available: false, space_id: 'space:kitchen',
      estimate: { validity: 'no_estimate', transform: null } },
    geometry: { box: { length_m: 0.32, width_m: 0.24, height_m: 0.18,
      center: { x: 0, y: 0, z: 0 } } },
    local_frame: { forward: '+x', left: '+y', up: '+z' },
  });
  const html = page([detail], detail.id, detail);
  assert.match(html, /Online/);
  assert.match(html, /Position currently unavailable/);
  assert.doesNotMatch(html, /telemetry-values/);
});

test('MacBook detail leaves unconfigured geometry and frame unknown', () => {
  const detail = body({
    id: 'embodiment:macbook-0', name: 'MacBook', embodiment_type: 'computer',
    geometry: null, local_frame: null,
    capabilities: [{ name: 'vision.observe', supported: true, available: false }],
  });
  const html = page([detail], detail.id, detail);
  assert.match(html, /MacBook/);
  assert.match(html, /Linked to: 老管家/);
  assert.match(html, /Offline/);
  assert.match(html, /Not configured/);
  assert.match(html, /vision\.observe/);
});

test('chat keeps selected offline body and separates it from agent identity', () => {
  assert.match(chat([base]), /老管家/);
  assert.match(chat([base]), /No embodiment/);
  const selected = chat([base], base.id);
  assert.match(selected, /老管家/);
  assert.match(selected, /via MicroDuck/);
  assert.match(selected, /MicroDuck Offline/);
  assert.doesNotMatch(selected, /MicroDuck<\/option>.*老管家/);
  const reconnected = chat([body({ connected: true, state: 'linked_online' })], base.id);
  assert.match(reconnected, /MicroDuck Online/);
  assert.match(chat([], base.id), /Selected body is no longer linked/);
});

test('unknown detail has a recovery path', () => {
  const html = page([], 'embodiment:deleted', null, 'not_found');
  assert.match(html, /Embodiment not found/);
  assert.match(html, /All embodiments/);
});
