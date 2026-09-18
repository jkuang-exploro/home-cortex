<script lang="ts">
  import { onMount } from 'svelte';
  import Chat from './components/Chat.svelte';
  import Login from './components/Login.svelte';
  import Sidebar from './components/Sidebar.svelte';
  import {
    createConversation,
    createSession,
    deleteConversation,
    deleteSession,
    getConversation,
    listConversations,
    listModels,
    readSession,
    streamMessage,
  } from './lib/api';
  import { detectLanguage, t } from './lib/i18n';
  import { ApiError, type ChatMessage, type ConversationSummary, type Language, type Model } from './lib/types';

  let language = $state<Language>(detectLanguage());
  let ready = $state(false);
  let needsLogin = $state(false);
  let loginError = $state('');
  let models = $state<Model[]>([]);
  let model = $state('');
  let conversations = $state<ConversationSummary[]>([]);
  let activeId = $state<string | null>(null);
  let messages = $state<ChatMessage[]>([]);
  let pending = $state(false);
  let sendError = $state('');
  let creating: Promise<string> | null = null;

  const copy = $derived(t(language));

  onMount(() => {
    void bootstrap();
  });

  async function bootstrap() {
    try {
      await readSession();
      await loadWorkspace();
      if (conversations[0]) await selectChat(conversations[0].id);
      else await startNewChat();
      needsLogin = false;
    } catch (error) {
      needsLogin = error instanceof ApiError && error.status === 401;
      if (!needsLogin) loginError = error instanceof Error ? error.message : String(error);
    } finally {
      ready = true;
    }
  }

  async function loadWorkspace() {
    models = await listModels();
    if (!model) {
      model = models.find((item) => item.kind !== 'model')?.id || models[0]?.id || '';
    }
    conversations = await listConversations();
  }

  async function login(email: string, apiKey: string) {
    loginError = '';
    try {
      await createSession(email, apiKey);
      await loadWorkspace();
      if (conversations[0]) await selectChat(conversations[0].id);
      else await startNewChat();
      needsLogin = false;
    } catch (error) {
      if (error instanceof ApiError && error.code === 'identity_not_mapped') {
        loginError = copy.mapped;
      } else if (error instanceof ApiError && error.status === 401) {
        loginError = copy.needKey;
      } else {
        loginError = error instanceof Error ? error.message : String(error);
      }
    }
  }

  async function ensureConversation(): Promise<string> {
    if (activeId) return activeId;
    if (creating) return creating;
    const startedFor = model;
    creating = createConversation(model, language)
      .then((conversation) => {
        const summary = {
          id: conversation.id,
          model: conversation.model,
          agent_id: conversation.agent_id,
          language: conversation.language,
          greeting: conversation.greeting,
          title: conversation.title,
          updated_at: conversation.updated_at,
        };
        conversations = [summary, ...conversations.filter((item) => item.id !== conversation.id)];
        if (!activeId && model === startedFor) {
          activeId = conversation.id;
          messages = conversation.messages ?? [];
        }
        return conversation.id;
      })
      .finally(() => {
        creating = null;
      });
    return creating;
  }

  async function startNewChat() {
    activeId = null;
    messages = [];
    creating = null;
    await ensureConversation();
  }

  async function selectChat(id: string) {
    const conversation = await getConversation(id);
    activeId = conversation.id;
    model = conversation.model;
    messages = conversation.messages ?? [];
  }

  async function removeChat(id: string) {
    await deleteConversation(id);
    conversations = conversations.filter((item) => item.id !== id);
    if (activeId === id) {
      activeId = null;
      messages = [];
    }
  }

  async function changeModel(next: string) {
    if (next === model) return;
    model = next;
    activeId = null;
    messages = [];
    creating = null;
    await ensureConversation();
  }

  async function send(text: string) {
    sendError = '';
    const user: ChatMessage = { id: crypto.randomUUID(), role: 'user', content: text };
    const assistant: ChatMessage = { id: crypto.randomUUID(), role: 'assistant', content: '' };
    messages = [...messages, user, assistant];
    pending = true;
    try {
      const conversationId = activeId ?? (await ensureConversation());
      if (!activeId) activeId = conversationId;
      for await (const delta of streamMessage(conversationId, text)) {
        assistant.content += delta;
        messages = [...messages];
      }
      if (!assistant.content) {
        assistant.content = sendError || 'No reply';
        messages = [...messages];
      }
      conversations = await listConversations();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      sendError = message;
      assistant.content = message;
      messages = [...messages];
    } finally {
      pending = false;
    }
  }

  async function signOut() {
    await deleteSession();
    needsLogin = true;
    conversations = [];
    messages = [];
    activeId = null;
  }
</script>

{#if !ready}
  <p class="empty">{copy.product}</p>
{:else if needsLogin}
  <Login {language} error={loginError} onsubmit={login} />
{:else}
  <div class="shell">
    <Sidebar
      {language}
      {conversations}
      {activeId}
      onnew={startNewChat}
      onselect={selectChat}
      ondelete={removeChat}
      onsignout={signOut}
    />
    <Chat
      {language}
      {models}
      {model}
      {messages}
      {pending}
      error={sendError}
      onmodel={changeModel}
      onlanguage={(next) => (language = next)}
      onsend={send}
    />
  </div>
{/if}
