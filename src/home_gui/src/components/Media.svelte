<script lang="ts">
  import { onMount } from 'svelte';
  import {
    listMedia,
    refreshMedia,
    type MediaFilter,
    type MediaItem,
  } from '../lib/media';

  let { onauthrequired }: { onauthrequired: () => void } = $props();

  let items = $state<MediaItem[]>([]);
  let filter = $state<MediaFilter>('all');
  let total = $state(0);
  let nextOffset = $state<number | null>(null);
  let loading = $state(false);
  let refreshing = $state(false);
  let error = $state('');
  let selectedIndex = $state<number | null>(null);

  const selected = $derived(selectedIndex === null ? null : items[selectedIndex] ?? null);
  const grouped = $derived(groupByMonth(items));

  onMount(() => {
    void load(true);
  });

  async function load(reset: boolean) {
    if (loading) return;
    loading = true;
    error = '';
    try {
      const page = await listMedia(filter, reset ? 0 : (nextOffset ?? 0));
      items = reset ? page.items : [...items, ...page.items];
      total = page.total;
      nextOffset = page.next_offset;
      if (reset) selectedIndex = null;
    } catch (failure) {
      if ((failure as { status?: number }).status === 401) onauthrequired();
      else error = failure instanceof Error ? failure.message : String(failure);
    } finally {
      loading = false;
    }
  }

  async function setFilter(next: MediaFilter) {
    if (next === filter) return;
    filter = next;
    await load(true);
  }

  async function refresh() {
    refreshing = true;
    error = '';
    try {
      await refreshMedia();
      await load(true);
    } catch (failure) {
      if ((failure as { status?: number }).status === 401) onauthrequired();
      else error = failure instanceof Error ? failure.message : String(failure);
    } finally {
      refreshing = false;
    }
  }

  function openItem(id: string) {
    const index = items.findIndex((item) => item.id === id);
    if (index >= 0) selectedIndex = index;
  }

  function move(delta: number) {
    if (selectedIndex === null) return;
    const next = selectedIndex + delta;
    if (next >= 0 && next < items.length) selectedIndex = next;
  }

  function handleKeydown(event: KeyboardEvent) {
    if (selectedIndex === null) return;
    if (event.key === 'Escape') selectedIndex = null;
    else if (event.key === 'ArrowLeft') move(-1);
    else if (event.key === 'ArrowRight') move(1);
  }

  function groupByMonth(source: MediaItem[]): { label: string; items: MediaItem[] }[] {
    const groups = new Map<string, { label: string; items: MediaItem[] }>();
    for (const item of source) {
      const date = new Date(item.captured_at);
      const key = `${date.getUTCFullYear()}-${String(date.getUTCMonth() + 1).padStart(2, '0')}`;
      const label = Number.isNaN(date.valueOf())
        ? 'Unknown date'
        : new Intl.DateTimeFormat(undefined, { month: 'long', year: 'numeric' }).format(date);
      const group = groups.get(key) ?? { label, items: [] };
      group.items.push(item);
      groups.set(key, group);
    }
    return [...groups.values()];
  }

  function formatDuration(seconds: number | null): string {
    if (seconds === null) return '';
    const rounded = Math.round(seconds);
    return `${Math.floor(rounded / 60)}:${String(rounded % 60).padStart(2, '0')}`;
  }

  function formatDate(value: string): string {
    const date = new Date(value);
    return Number.isNaN(date.valueOf()) ? value : date.toLocaleString();
  }
</script>

<svelte:window onkeydown={handleKeydown} />

<main class="media-main">
  <header class="media-bar">
    <div>
      <p class="eyebrow">HOUSEHOLD LIBRARY</p>
      <h2>Media</h2>
      <p class="media-count">{total} {filter === 'all' ? 'items' : filter === 'photo' ? 'photos' : 'videos'}</p>
    </div>
    <button class="ghost compact" type="button" onclick={refresh} disabled={refreshing || loading}>
      {refreshing ? 'Refreshing…' : 'Refresh library'}
    </button>
  </header>

  <nav class="media-filters" aria-label="Media type">
    {#each [['all', 'All'], ['photo', 'Photos'], ['video', 'Videos']] as option}
      <button
        type="button"
        class:active={filter === option[0]}
        aria-pressed={filter === option[0]}
        onclick={() => setFilter(option[0] as MediaFilter)}
      >{option[1]}</button>
    {/each}
  </nav>

  <section class="media-scroll">
    {#if error}
      <div class="media-state error-panel">
        <h3>Media is unavailable</h3>
        <p>{error}</p>
        <button class="primary" type="button" onclick={() => load(true)}>Try again</button>
      </div>
    {:else if loading && items.length === 0}
      <div class="media-state"><p>Loading your library…</p></div>
    {:else if items.length === 0}
      <div class="media-state">
        <h3>No media indexed yet</h3>
        <p>Refresh the library to discover photos and videos in the read-only source folders.</p>
        <button class="primary" type="button" onclick={refresh} disabled={refreshing}>Refresh library</button>
      </div>
    {:else}
      {#each grouped as group (group.label)}
        <section class="media-month">
          <h3>{group.label}</h3>
          <div class="media-grid">
            {#each group.items as item (item.id)}
              <button
                class="media-tile"
                type="button"
                aria-label={`Open ${item.filename}`}
                onclick={() => openItem(item.id)}
              >
                <img src={item.thumbnail_url} alt={item.filename} loading="lazy" />
                <span class="tile-shade"></span>
                {#if item.type === 'video'}
                  <span class="video-mark" aria-hidden="true">▶</span>
                  {#if item.duration_seconds !== null}
                    <span class="duration">{formatDuration(item.duration_seconds)}</span>
                  {/if}
                {/if}
                {#if item.metadata_status !== 'ok'}
                  <span class="media-warning" title="Some metadata could not be read">!</span>
                {/if}
              </button>
            {/each}
          </div>
        </section>
      {/each}
      {#if nextOffset !== null}
        <div class="load-more">
          <button class="ghost compact" type="button" onclick={() => load(false)} disabled={loading}>
            {loading ? 'Loading…' : 'Load more'}
          </button>
        </div>
      {/if}
    {/if}
  </section>
</main>

{#if selected}
  <div class="media-viewer" role="dialog" aria-modal="true" aria-label={selected.filename}>
    <button class="viewer-close" type="button" aria-label="Close" onclick={() => (selectedIndex = null)}>×</button>
    <button class="viewer-nav previous" type="button" aria-label="Previous" onclick={() => move(-1)} disabled={selectedIndex === 0}>‹</button>
    <div class="viewer-content">
      {#if selected.type === 'photo'}
        <img src={selected.content_url} alt={selected.filename} />
      {:else}
        <!-- svelte-ignore a11y_media_has_caption -->
        <video src={selected.content_url} poster={selected.thumbnail_url} controls autoplay></video>
      {/if}
      <div class="viewer-meta">
        <strong>{selected.filename}</strong>
        <span>{formatDate(selected.captured_at)}</span>
        {#if selected.width && selected.height}<span>{selected.width} × {selected.height}</span>{/if}
        {#if selected.camera_make || selected.camera_model}
          <span>{[selected.camera_make, selected.camera_model].filter(Boolean).join(' ')}</span>
        {/if}
      </div>
    </div>
    <button class="viewer-nav next" type="button" aria-label="Next" onclick={() => move(1)} disabled={selectedIndex === items.length - 1}>›</button>
  </div>
{/if}
