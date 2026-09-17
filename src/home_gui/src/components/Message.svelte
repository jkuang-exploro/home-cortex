<script lang="ts">
  import { renderMarkdown } from '../lib/markdown';
  import type { ChatMessage } from '../lib/types';

  let { message, pending = false }: { message: ChatMessage; pending?: boolean } = $props();
  const html = $derived(renderMarkdown(message.content || (pending ? '…' : '')));
</script>

<article class="bubble {message.role}">
  <p class="who">{message.role === 'user' ? 'You' : 'Cortex'}</p>
  <div>{@html html}</div>
</article>
