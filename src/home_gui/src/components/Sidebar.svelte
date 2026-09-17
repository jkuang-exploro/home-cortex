<script lang="ts">
  import { t } from '../lib/i18n';
  import type { ConversationSummary, Language } from '../lib/types';

  let {
    language,
    conversations,
    activeId,
    onnew,
    onselect,
    ondelete,
    onsignout,
  }: {
    language: Language;
    conversations: ConversationSummary[];
    activeId: string | null;
    onnew: () => void;
    onselect: (id: string) => void;
    ondelete: (id: string) => void;
    onsignout: () => void;
  } = $props();

  const copy = $derived(t(language));
</script>

<aside class="sidebar">
  <div class="brand">
    <p>HOME CORTEX</p>
    <h1>{copy.tagline}</h1>
  </div>
  <div class="sidebar-actions">
    <button class="primary" type="button" onclick={onnew}>{copy.newChat}</button>
    <a class="ghost" href="/vision">{copy.vision}</a>
  </div>
  <div class="chat-list" aria-label={copy.chats}>
    {#each conversations as chat (chat.id)}
      <div class="chat-item" class:active={chat.id === activeId}>
        <button type="button" onclick={() => onselect(chat.id)}>
          {chat.title || copy.empty}
        </button>
        <button
          class="danger"
          type="button"
          aria-label={copy.delete}
          onclick={() => ondelete(chat.id)}
        >×</button>
      </div>
    {/each}
  </div>
  <div class="sidebar-foot">
    <button class="ghost" type="button" onclick={onsignout}>{copy.signOut}</button>
  </div>
</aside>
