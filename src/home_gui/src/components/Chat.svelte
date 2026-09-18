<script lang="ts">
  import { t } from '../lib/i18n';
  import type { ChatMessage, Language, Model } from '../lib/types';
  import Composer from './Composer.svelte';
  import Message from './Message.svelte';

  let {
    language,
    models,
    model,
    messages,
    pending = false,
    error = '',
    onmodel,
    onlanguage,
    onsend,
  }: {
    language: Language;
    models: Model[];
    model: string;
    messages: ChatMessage[];
    pending?: boolean;
    error?: string;
    onmodel: (id: string) => void;
    onlanguage: (language: Language) => void;
    onsend: (text: string) => void;
  } = $props();

  const copy = $derived(t(language));
  const agents = $derived(models.filter((item) => item.kind !== 'model'));
  const bare = $derived(models.filter((item) => item.kind === 'model'));
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
  <div class="messages">
    {#if messages.length === 0}
      <p class="empty">{copy.empty}</p>
    {:else}
      {#each messages as message (message.id)}
        <Message
          {message}
          pending={pending && message.role === 'assistant' && message === messages[messages.length - 1] && !message.content}
          thinking={copy.thinking}
        />
      {/each}
    {/if}
  </div>
  {#if error}<p class="error">{error}</p>{/if}
  {#if pending}<p class="help">{copy.thinking}</p>{/if}
  <Composer {language} disabled={pending} {onsend} />
</section>
