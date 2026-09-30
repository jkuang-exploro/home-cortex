<script lang="ts">
  import { t } from '../lib/i18n';
  import { capabilityLabel, dateTime, physicalValue, relativeTime, spaceLabel } from '../lib/embodiments';
  import type { EmbodimentDetail, EmbodimentSummary, Language } from '../lib/types';

  let {
    language, items, detail, selectedId, loading, error, detailError,
    onselect, onback, onrefresh,
  }: {
    language: Language;
    items: EmbodimentSummary[];
    detail: EmbodimentDetail | null;
    selectedId: string | null;
    loading: boolean;
    error: string;
    detailError: string;
    onselect: (id: string) => void;
    onback: () => void;
    onrefresh: () => void;
  } = $props();

  const copy = $derived(t(language));
  const positions = ['x', 'y', 'z'] as const;
  const orientations = ['yaw', 'pitch', 'roll'] as const;
</script>

<main class="embodiments-main">
  <header class="embodiments-heading">
    <div>
      <p class="eyebrow">HOME CORTEX</p>
      <h2>{copy.embodiments}</h2>
      <p>{copy.embodimentsHelp}</p>
    </div>
    <button class="ghost compact" type="button" onclick={onrefresh} disabled={loading}>
      {copy.refreshBodies}
    </button>
  </header>
  <div class="embodiments-scroll">
    {#if error}
      <div class="embodiment-empty error-panel" role="alert">
        <h3>{copy.embodimentsUnavailable}</h3>
        <p>{error}</p>
        <button class="primary" type="button" onclick={onrefresh}>{copy.tryAgain}</button>
      </div>
    {:else if selectedId}
      <button class="embodiment-back" type="button" onclick={onback}>← {copy.allEmbodiments}</button>
      {#if detailError}
        <div class="embodiment-empty" role="status">
          <h3>{detailError === 'not_found' ? copy.embodimentMissing : copy.embodimentsUnavailable}</h3>
          <p>{detailError === 'not_found' ? selectedId : detailError}</p>
          {#if detailError !== 'not_found'}
            <button class="primary" type="button" onclick={onrefresh}>{copy.tryAgain}</button>
          {/if}
        </div>
      {:else if !detail}
        <div class="embodiment-empty" role="status">{copy.loadingBodies}</div>
      {:else}
        <article class="embodiment-detail">
          <div class="embodiment-detail-head">
            <div>
              <h3>{detail.name}</h3>
              <code>{detail.id}</code>
            </div>
            <span class="presence-pill" class:online={detail.connected}>
              <span aria-hidden="true">{detail.connected ? '●' : '○'}</span>
              {detail.connected ? copy.online : copy.offline}
            </span>
          </div>
          <p class="embodiment-association">
            {detail.agent ? `${copy.linkedTo}: ${detail.agent.name ?? detail.agent.id}` : copy.notLinked}
          </p>

          <div class="embodiment-detail-grid">
            <section class="embodiment-section">
              <h4>{copy.runtimePresence}</h4>
              <dl>
                <dt>{copy.connection}</dt><dd>{detail.connected ? copy.online : copy.offline}</dd>
                <dt>{copy.connectedAt}</dt><dd>{dateTime(detail.runtime.connected_at, language)}</dd>
                <dt>{copy.lastSeen}</dt>
                <dd>{detail.runtime.last_seen
                  ? `${relativeTime(detail.runtime.last_seen, language)} · ${dateTime(detail.runtime.last_seen, language)}`
                  : copy.neverSeen}</dd>
              </dl>
            </section>
            <section class="embodiment-section">
              <h4>{copy.bodyGeometry}</h4>
              <dl>
                <dt>{copy.dimensions}</dt>
                <dd>{detail.geometry.box.length_m} × {detail.geometry.box.width_m} × {detail.geometry.box.height_m} m</dd>
                <dt>{copy.centerOffset}</dt>
                <dd>{detail.geometry.box.center.x}, {detail.geometry.box.center.y}, {detail.geometry.box.center.z} m</dd>
                <dt>{copy.localFrame}</dt>
                <dd>{copy.forward} {detail.local_frame.forward} · {copy.left} {detail.local_frame.left} · {copy.up} {detail.local_frame.up}</dd>
              </dl>
            </section>
            <section class="embodiment-section">
              <h4>{copy.capabilities}</h4>
              {#if detail.capabilities.length}
                <ul class="capability-list">
                  {#each detail.capabilities as capability (capability.name)}
                    <li>
                      <div><strong>{capabilityLabel(capability.name)}</strong><code>{capability.name}</code></div>
                      <span>{capability.supported ? copy.supported : copy.notSupported}</span>
                      <span class:capability-ready={capability.available}>
                        {capability.available ? copy.available : copy.unavailable}
                      </span>
                    </li>
                  {/each}
                </ul>
              {:else}
                <p class="quiet">{copy.noCapabilities}</p>
              {/if}
            </section>
            <section class="embodiment-section">
              <h4>{copy.physicalTelemetry}</h4>
              <p class:telemetry-ready={detail.telemetry.available} class="telemetry-status">
                {detail.telemetry.available ? copy.positionAvailable : copy.positionUnavailable}
              </p>
              <dl>
                <dt>{copy.space}</dt><dd>{detail.telemetry.space_id ? spaceLabel(detail.telemetry.space_id) : '—'}</dd>
                <dt>{copy.measuredAt}</dt><dd>{dateTime(detail.telemetry.measured_at, language)}</dd>
              </dl>
              {#if detail.telemetry.available && detail.telemetry.estimate?.transform}
                <div class="telemetry-values">
                  <div>
                    <h5>{copy.position}</h5>
                    {#each positions as axis}
                      <p><span>{axis}</span><span>{physicalValue(detail.telemetry.estimate.transform[axis], 'm')}</span></p>
                    {/each}
                  </div>
                  <div>
                    <h5>{copy.orientation}</h5>
                    {#each orientations as axis}
                      <p><span>{axis}</span><span>{physicalValue(detail.telemetry.estimate.transform[axis], 'rad')}</span></p>
                    {/each}
                  </div>
                </div>
                <p class="quiet">{copy.uncertaintyHelp}</p>
              {/if}
            </section>
          </div>
        </article>
      {/if}
    {:else if loading && items.length === 0}
      <div class="embodiment-empty" role="status">{copy.loadingBodies}</div>
    {:else if items.length === 0}
      <div class="embodiment-empty" role="status">
        <h3>{copy.noEmbodiments}</h3>
        <p>{copy.noEmbodimentsHelp}</p>
      </div>
    {:else}
      <div class="embodiment-grid">
        {#each items as body (body.id)}
          <button class="embodiment-card" type="button" onclick={() => onselect(body.id)}>
            <div class="embodiment-card-head">
              <h3>{body.name}</h3>
              <span class="presence-pill" class:online={body.connected}>
                <span aria-hidden="true">{body.connected ? '●' : '○'}</span>
                {body.connected ? copy.online : copy.offline}
              </span>
            </div>
            <p class="embodiment-association">
              {body.agent ? `${copy.linkedTo}: ${body.agent.name ?? body.agent.id}` : copy.notLinked}
            </p>
            <p class="embodiment-card-pose">
              {body.telemetry.available
                ? `${spaceLabel(body.telemetry.space_id)} · ${copy.positionAvailable}`
                : copy.positionUnavailable}
            </p>
            <p class="quiet">{copy.lastSeen}: {body.runtime.last_seen
              ? relativeTime(body.runtime.last_seen, language) : copy.neverSeen}</p>
            <div class="embodiment-card-caps">
              {#each body.capabilities as capability (capability.name)}
                <span class:capability-ready={capability.available}>
                  {capability.available ? '✓' : '○'} {capabilityLabel(capability.name)}
                </span>
              {/each}
            </div>
          </button>
        {/each}
      </div>
    {/if}
  </div>
</main>
