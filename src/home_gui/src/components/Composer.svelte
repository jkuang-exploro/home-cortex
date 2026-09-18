<script lang="ts">
  import { t } from '../lib/i18n';
  import type { Language } from '../lib/types';

  let { language, disabled = false, onsend }: {
    language: Language;
    disabled?: boolean;
    onsend: (text: string) => void;
  } = $props();

  const copy = $derived(t(language));
  let draft = $state('');
  let field: HTMLTextAreaElement | undefined;

  $effect(() => {
    if (disabled) return;
    requestAnimationFrame(() => field?.focus());
  });

  function send() {
    const text = draft.trim();
    if (!text || disabled) return;
    draft = '';
    onsend(text);
  }

  function onkeydown(event: KeyboardEvent) {
    if (event.key !== 'Enter' || event.shiftKey) return;
    if (event.isComposing || event.keyCode === 229) return;
    event.preventDefault();
    send();
  }
</script>

<div class="composer">
  <div class="composer-box">
    <textarea
      bind:this={field}
      bind:value={draft}
      placeholder={copy.placeholder}
      {disabled}
      onkeydown={onkeydown}
      rows="2"
    ></textarea>
    <button class="primary" type="button" {disabled} onclick={send}>{copy.send}</button>
  </div>
</div>
