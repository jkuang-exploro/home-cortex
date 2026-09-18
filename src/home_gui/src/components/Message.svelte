<script lang="ts">
  import { renderMarkdown } from '../lib/markdown';
  import type { ChatMessage } from '../lib/types';

  let {
    message,
    pending = false,
    thinking = '…',
    speaker = 'You',
    assistant = '老管家',
  }: {
    message: ChatMessage;
    pending?: boolean;
    thinking?: string;
    speaker?: string;
    assistant?: string;
  } = $props();
  const html = $derived(renderMarkdown(message.content || (pending ? thinking : '')));
</script>

<article class="bubble {message.role}">
  <p class="who">{message.role === 'user' ? speaker : assistant}</p>
  <div>{@html html}</div>
</article>
