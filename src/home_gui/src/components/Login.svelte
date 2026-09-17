<script lang="ts">
  import { t } from '../lib/i18n';
  import type { Language } from '../lib/types';

  let { language, error = '', onsubmit }: {
    language: Language;
    error?: string;
    onsubmit: (email: string, apiKey: string) => void;
  } = $props();

  const copy = $derived(t(language));
  let email = $state('');
  let apiKey = $state('');

  function submit(event: SubmitEvent) {
    event.preventDefault();
    onsubmit(email.trim(), apiKey);
    apiKey = '';
  }
</script>

<main class="login">
  <form onsubmit={submit}>
    <p>{copy.product}</p>
    <h1>{copy.tagline}</h1>
    <label>
      {copy.email}
      <input type="email" bind:value={email} autocomplete="username" required />
    </label>
    <label>
      {copy.apiKey}
      <input type="password" bind:value={apiKey} autocomplete="current-password" required />
    </label>
    {#if error}<p class="error">{error}</p>{/if}
    <button class="primary" type="submit">{copy.signIn}</button>
    <p class="help">{copy.loginHelp}</p>
  </form>
</main>
