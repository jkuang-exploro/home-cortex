<script lang="ts">
  import { t } from '../lib/i18n';
  import { spaceLabel } from '../lib/embodiments';
  import type { ChatMessage, EmbodimentSummary, Language, Model } from '../lib/types';
  import Composer from './Composer.svelte';
  import Message from './Message.svelte';

  let {
    language,
    models,
    model,
    messages,
    pending = false,
    embodiments,
    activeEmbodimentId,
    activeAgentId,
    activeAgentEntityId,
    embodimentError = '',
    error = '',
    onmodel,
    onlanguage,
    onsend,
    onembodiment,
    onrefresh,
    onopenbody,
  }: {
    language: Language;
    models: Model[];
    model: string;
    messages: ChatMessage[];
    pending?: boolean;
    embodiments: EmbodimentSummary[];
    activeEmbodimentId: string | null;
    activeAgentId: string | null;
    activeAgentEntityId: string | null;
    embodimentError?: string;
    error?: string;
    onmodel: (id: string) => void;
    onlanguage: (language: Language) => void;
    onsend: (text: string) => void;
    onembodiment: (id: string | null) => void;
    onrefresh: () => void;
    onopenbody: (id: string) => void;
  } = $props();

  const copy = $derived(t(language));
  const agents = $derived(models.filter((item) => item.kind !== 'model'));
  const bare = $derived(models.filter((item) => item.kind === 'model'));
  const selectedBody = $derived(embodiments.find((item) => item.id === activeEmbodimentId));
  const linkedBodies = $derived(embodiments.filter((item) => item.agent?.id === activeAgentEntityId));

  let scroller: HTMLDivElement | undefined;

  $effect(() => {
    messages.length;
    messages.at(-1)?.content;
    pending;
    requestAnimationFrame(() => {
      if (!scroller) return;
      scroller.scrollTop = scroller.scrollHeight;
    });
  });
</script>

<section class="main">
  <header class="main-bar">
    <label>
      {copy.models}
      <select value={model} onchange={(event) => onmodel((event.currentTarget as HTMLSelectElement).value)}>
        {#if agents.length}
          <optgroup label={copy.agents}>
            {#each agents as item}<option value={item.id}>{item.id}</option>{/each}
          </optgroup>
        {/if}
        {#if bare.length}
          <optgroup label={copy.bare}>
            {#each bare as item}<option value={item.id}>{item.id}</option>{/each}
          </optgroup>
        {/if}
      </select>
    </label>
    <select
      aria-label="language"
      value={language}
      onchange={(event) => onlanguage((event.currentTarget as HTMLSelectElement).value as Language)}
    >
      <option value="zh">中文</option>
      <option value="en">English</option>
    </select>
  </header>
  {#if activeAgentId}
    <div class="embodiment-bar">
      <strong class="chat-agent-context">
        {model}
        {#if activeEmbodimentId}
          <span>{copy.via} {selectedBody?.name ?? activeEmbodimentId}</span>
        {:else}
          <span>· {copy.noBody}</span>
        {/if}
      </strong>
      <label>
        {copy.body}
        <select
          aria-label={copy.body}
          value={activeEmbodimentId ?? ''}
          disabled={pending}
          onchange={(event) => onembodiment((event.currentTarget as HTMLSelectElement).value || null)}
        >
          <option value="">{copy.noBody}</option>
          {#if activeEmbodimentId && (!selectedBody || selectedBody.agent?.id !== activeAgentEntityId)}
            <option value={activeEmbodimentId} disabled>{activeEmbodimentId} · {copy.bodyUnavailable}</option>
          {/if}
          {#each linkedBodies as body (body.id)}
            <option value={body.id}>
              {body.name} — {body.connected ? copy.online : copy.offline}
            </option>
          {/each}
        </select>
      </label>
      {#if selectedBody && selectedBody.agent?.id === activeAgentEntityId}
        <span class:online={selectedBody.connected} class="body-state">
          {selectedBody.connected ? '●' : '○'} {selectedBody.name} {selectedBody.connected ? copy.online : copy.offline}
        </span>
        {#if selectedBody.telemetry.available}
          <span class="body-detail">{spaceLabel(selectedBody.telemetry.space_id)}</span>
        {/if}
        <button class="body-detail-link" type="button" onclick={() => onopenbody(selectedBody.id)}>
          {copy.details}
        </button>
      {:else if activeEmbodimentId}
        <span class="body-state">{copy.bodyUnavailable}</span>
      {:else if !linkedBodies.length}
        <span class="body-detail">{copy.noLinkedBodies}</span>
      {/if}
      <button class="body-refresh" type="button" onclick={onrefresh} aria-label={copy.refreshBodies}>{copy.refreshBodies}</button>
    </div>
    {#if embodimentError}<p class="error embodiment-error">{embodimentError}</p>{/if}
  {/if}
  <div class="messages" bind:this={scroller}>
    {#if messages.length === 0}
      <p class="empty">{copy.empty}</p>
    {:else}
      {#each messages as message (message.id)}
        <Message
          {message}
          pending={pending && message.role === 'assistant' && message === messages[messages.length - 1] && !message.content}
          thinking={copy.thinking}
          speaker={copy.you}
          assistant={model || copy.tagline}
        />
      {/each}
    {/if}
  </div>
  {#if error}<p class="error">{error}</p>{/if}
  {#if pending}<p class="help">{copy.thinking}</p>{/if}
  <Composer {language} disabled={pending} {onsend} />
</section>
