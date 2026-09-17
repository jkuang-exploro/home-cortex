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

  function submit(event: SubmitEvent) {
    event.preventDefault();
    const text = draft.trim();
    if (!text || disabled) return;
    onsend(text);
    draft = '';
  }

  function onkeydown(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      (event.currentTarget as HTMLTextAreaElement).form?.requestSubmit();
    }
  }
</script>

<div class="composer">
  <form onsubmit={submit}>
    <textarea
      bind:value={draft}
      placeholder={copy.placeholder}
      {disabled}
      onkeydown={onkeydown}
      rows="2"
    ></textarea>
    <button class="primary" type="submit" {disabled}>{copy.send}</button>
  </form>
</div>
