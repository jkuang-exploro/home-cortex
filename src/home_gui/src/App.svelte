<script lang="ts">
  import { onMount } from 'svelte';
  import Chat from './components/Chat.svelte';
  import Embodiments from './components/Embodiments.svelte';
  import Login from './components/Login.svelte';
  import Media from './components/Media.svelte';
  import Sidebar from './components/Sidebar.svelte';
  import {
    createConversation,
    createSession,
    deleteConversation,
    deleteSession,
    getConversation,
    getEmbodiment,
    listEmbodiments,
    listConversations,
    listModels,
    readSession,
    setActiveEmbodiment,
    streamMessage,
  } from './lib/api';
  import { detectLanguage, t } from './lib/i18n';
  import { pathForRoute, routeFromPath, type Page } from './lib/navigation';
  import { ApiError, type ChatMessage, type ConversationSummary, type EmbodimentDetail, type EmbodimentSummary, type Language, type Model } from './lib/types';

  const initialRoute = routeFromPath(window.location.pathname);

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
  let embodimentError = $state('');
  let embodiments = $state<EmbodimentSummary[]>([]);
  let embodimentDetail = $state<EmbodimentDetail | null>(null);
  let embodimentDetailError = $state('');
  let embodimentsLoading = $state(false);
  let selectedEmbodimentId = $state<string | null>(initialRoute.embodimentId);
  let activeEmbodimentId = $state<string | null>(null);
  let activeAgentId = $state<string | null>(null);
  let activeAgentEntityId = $state<string | null>(null);
  let embodimentRequest = 0;
  let embodimentFetches = 0;
  let creating: Promise<string> | null = null;
  let page = $state<Page>(initialRoute.page);

  const copy = $derived(t(language));

  function newId(): string {
    try {
      if (globalThis.crypto?.randomUUID) return crypto.randomUUID();
    } catch {
      /* http origins are not a secure context */
    }
    return `${Date.now().toString(16)}-${Math.random().toString(16).slice(2)}`;
  }

  onMount(() => {
    const handlePopState = () => void navigate(routeFromPath(window.location.pathname), false);
    window.addEventListener('popstate', handlePopState);
    void bootstrap();
    const poll = window.setInterval(() => {
      if (ready && !needsLogin && (page === 'embodiments' || (page === 'chat' && activeAgentId))) {
        void refreshEmbodiments(true);
      }
    }, 1500);
    return () => {
      window.removeEventListener('popstate', handlePopState);
      window.clearInterval(poll);
    };
  });

  async function bootstrap() {
    try {
      await readSession();
      if (page === 'chat') await openWorkspace();
      if (page === 'embodiments') await refreshEmbodiments();
      needsLogin = false;
    } catch (error) {
      needsLogin = error instanceof ApiError && error.status === 401;
      if (!needsLogin) loginError = error instanceof Error ? error.message : String(error);
    } finally {
      ready = true;
    }
  }

  async function loadWorkspace() {
    const [nextModels, nextConversations] = await Promise.all([
      listModels(),
      listConversations(),
    ]);
    models = nextModels;
    if (!model) {
      model = models.find((item) => item.kind !== 'model')?.id || models[0]?.id || '';
    }
    conversations = nextConversations;
  }

  async function openWorkspace() {
    await loadWorkspace();
    if (conversations[0]) await selectChat(conversations[0].id);
    else await startNewChat();
  }

  async function navigate(route: { page: Page; embodimentId: string | null },
                          push = true, openChat = true) {
    if (push) history.pushState({}, '', pathForRoute(route));
    embodimentRequest += 1;
    embodimentsLoading = false;
    page = route.page;
    selectedEmbodimentId = route.embodimentId;
    embodimentDetail = null;
    embodimentDetailError = '';
    if (openChat && route.page === 'chat' && conversations.length === 0) await openWorkspace();
    if (route.page === 'embodiments') await refreshEmbodiments();
  }

  async function showPage(next: Page) {
    await navigate({ page: next, embodimentId: null });
  }

  async function openEmbodiment(id: string) {
    await navigate({ page: 'embodiments', embodimentId: id });
  }

  async function newChatFromAnywhere() {
    if (page !== 'chat') {
      await navigate({ page: 'chat', embodimentId: null }, true, false);
      await loadWorkspace();
    }
    await startNewChat();
  }

  async function login(email: string, apiKey: string) {
    loginError = '';
    try {
      await createSession(email, apiKey);
      if (page === 'chat') await openWorkspace();
      if (page === 'embodiments') await refreshEmbodiments();
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
        const summary = conversationSummary(conversation);
        conversations = [summary, ...conversations.filter((item) => item.id !== conversation.id)];
        if (!activeId && model === startedFor) {
          activeId = conversation.id;
          messages = conversation.messages ?? [];
          activeEmbodimentId = conversation.active_embodiment_id;
          activeAgentId = conversation.agent_id;
          activeAgentEntityId = conversation.agent_entity_id ?? null;
          void refreshEmbodiments();
        }
        return conversation.id;
      })
      .finally(() => {
        creating = null;
      });
    return creating;
  }

  function conversationSummary(conversation: ConversationSummary): ConversationSummary {
    return {
      id: conversation.id,
      model: conversation.model,
      agent_id: conversation.agent_id,
      agent_entity_id: conversation.agent_entity_id,
      active_embodiment_id: conversation.active_embodiment_id,
      language: conversation.language,
      greeting: conversation.greeting,
      title: conversation.title,
      updated_at: conversation.updated_at,
    };
  }

  function promoteConversation(conversation: ConversationSummary) {
    conversations = [
      conversationSummary(conversation),
      ...conversations.filter((item) => item.id !== conversation.id),
    ];
  }

  async function startNewChat() {
    embodimentRequest += 1;
    activeId = null;
    messages = [];
    activeEmbodimentId = null;
    activeAgentId = null;
    activeAgentEntityId = null;
    embodiments = [];
    embodimentDetail = null;
    selectedEmbodimentId = null;
    creating = null;
    await ensureConversation();
  }

  async function selectChat(id: string) {
    const conversation = await getConversation(id);
    embodimentRequest += 1;
    embodiments = [];
    activeId = conversation.id;
    model = conversation.model;
    messages = conversation.messages ?? [];
    activeEmbodimentId = conversation.active_embodiment_id;
    activeAgentId = conversation.agent_id;
    activeAgentEntityId = conversation.agent_entity_id ?? null;
    void refreshEmbodiments();
  }

  async function refreshEmbodiments(silent = false) {
    if (silent && embodimentFetches > 0) return;
    embodimentFetches += 1;
    const agentId = activeAgentId;
    const requestNumber = ++embodimentRequest;
    const detailId = page === 'embodiments' ? selectedEmbodimentId : null;
    if (!silent) {
      embodimentsLoading = true;
      embodimentError = '';
    }
    if (page === 'chat' && !agentId) {
      embodiments = [];
      embodimentsLoading = false;
      embodimentFetches -= 1;
      return;
    }
    try {
      const next = await listEmbodiments();
      if (requestNumber !== embodimentRequest) return;
      embodiments = next;
      embodimentError = '';
      if (detailId) {
        try {
          const nextDetail = await getEmbodiment(detailId);
          if (requestNumber === embodimentRequest) {
            embodimentDetail = nextDetail;
            embodimentDetailError = '';
          }
        } catch (error) {
          if (requestNumber === embodimentRequest) {
            embodimentDetail = null;
            embodimentDetailError = error instanceof ApiError && error.status === 404
              ? 'not_found' : error instanceof Error ? error.message : String(error);
            if (error instanceof ApiError && error.status === 401) needsLogin = true;
          }
        }
      }
    } catch (error) {
      if (requestNumber === embodimentRequest) {
        embodimentError = error instanceof Error ? error.message : String(error);
        if (error instanceof ApiError && error.status === 401) needsLogin = true;
      }
    } finally {
      embodimentFetches -= 1;
      if (requestNumber === embodimentRequest) embodimentsLoading = false;
    }
  }

  async function selectEmbodiment(next: string | null) {
    if (!activeId || pending) return;
    const conversationId = activeId;
    embodimentError = '';
    try {
      const updated = await setActiveEmbodiment(conversationId, next);
      if (activeId !== conversationId) return;
      activeEmbodimentId = updated.active_embodiment_id;
      promoteConversation(updated);
      void refreshEmbodiments();
    } catch (error) {
      if (activeId !== conversationId) return;
      embodimentError = error instanceof Error ? error.message : String(error);
    }
  }

  async function removeChat(id: string) {
    await deleteConversation(id);
    conversations = conversations.filter((item) => item.id !== id);
    if (activeId === id) {
      embodimentRequest += 1;
      activeId = null;
      messages = [];
      activeEmbodimentId = null;
      activeAgentId = null;
      activeAgentEntityId = null;
      embodiments = [];
    }
  }

  async function changeModel(next: string) {
    if (next === model) return;
    model = next;
    embodimentRequest += 1;
    activeId = null;
    messages = [];
    activeEmbodimentId = null;
    activeAgentId = null;
    activeAgentEntityId = null;
    embodiments = [];
    creating = null;
    await ensureConversation();
  }

  function setLastAssistant(content: string) {
    const last = messages.at(-1);
    if (!last || last.role !== 'assistant') {
      messages = [...messages, { id: newId(), role: 'assistant', content }];
      return;
    }
    messages = [...messages.slice(0, -1), { ...last, content }];
  }

  async function send(text: string) {
    let profile = false;
    try {
      profile = localStorage.getItem('cortex:profile-first-answer') === '1';
    } catch {
      // Storage can be unavailable in a restricted browser context.
    }
    const started = performance.now();
    const timing: Record<string, number> = {};
    let requestId: string | undefined;
    const record = (event: string, id?: string) => {
      if (!profile) return;
      timing[event] ??= Math.round((performance.now() - started) * 10) / 10;
      if (id) requestId = id;
    };
    record('send');
    sendError = '';
    messages = [
      ...messages,
      { id: newId(), role: 'user', content: text },
      { id: newId(), role: 'assistant', content: '' },
    ];
    pending = true;
    if (profile) requestAnimationFrame(() => requestAnimationFrame(() => record('pending_paint')));
    let firstPaintScheduled = false;
    try {
      const conversationId = activeId ?? (await ensureConversation());
      if (!activeId) activeId = conversationId;
      record('conversation_ready');
      let reply = '';
      let paintFrame = 0;
      const paint = () => {
        paintFrame = 0;
        setLastAssistant(reply);
        if (profile && !firstPaintScheduled) {
          firstPaintScheduled = true;
          requestAnimationFrame(() => requestAnimationFrame(() =>
            record('first_content_paint')
          ));
        }
      };
      for await (const delta of streamMessage(conversationId, text, record)) {
        reply += delta;
        if (!paintFrame) paintFrame = requestAnimationFrame(paint);
      }
      if (paintFrame) cancelAnimationFrame(paintFrame);
      setLastAssistant(reply);
      if (profile && reply) requestAnimationFrame(() => requestAnimationFrame(() =>
        record('answer_complete_paint')
      ));
      if (profile && reply && !firstPaintScheduled) {
        firstPaintScheduled = true;
        requestAnimationFrame(() => requestAnimationFrame(() =>
          record('first_content_paint')
        ));
      }
      const latest = await getConversation(conversationId);
      if (latest.messages?.length) messages = latest.messages;
      else if (!reply) setLastAssistant(sendError || 'No reply');
      promoteConversation(latest);
      activeEmbodimentId = latest.active_embodiment_id;
      void refreshEmbodiments();
    } catch (error) {
      record('error');
      const message = error instanceof Error ? error.message : String(error);
      sendError = message;
      setLastAssistant(message);
    } finally {
      pending = false;
      if (profile) requestAnimationFrame(() => requestAnimationFrame(() => {
        console.info('first_answer_profile', { request_id: requestId, timing });
      }));
    }
  }

  async function signOut() {
    await deleteSession();
    embodimentRequest += 1;
    needsLogin = true;
    conversations = [];
    messages = [];
    activeId = null;
    activeEmbodimentId = null;
    activeAgentId = null;
    activeAgentEntityId = null;
    embodiments = [];
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
      {page}
      onnew={newChatFromAnywhere}
      onchat={() => void showPage('chat')}
      onmedia={() => void showPage('media')}
      onembodiments={() => void showPage('embodiments')}
      onselect={selectChat}
      ondelete={removeChat}
      onsignout={signOut}
    />
    {#if page === 'media'}
      <Media onauthrequired={() => (needsLogin = true)} />
    {:else if page === 'embodiments'}
      <Embodiments
        {language}
        items={embodiments}
        detail={embodimentDetail}
        selectedId={selectedEmbodimentId}
        loading={embodimentsLoading}
        error={embodimentError}
        detailError={embodimentDetailError}
        onselect={(id) => void openEmbodiment(id)}
        onback={() => void showPage('embodiments')}
        onrefresh={() => void refreshEmbodiments()}
      />
    {:else}
      <Chat
        {language}
        {models}
        {model}
        {messages}
        {pending}
        {embodiments}
        {activeEmbodimentId}
        {activeAgentId}
        {activeAgentEntityId}
        {embodimentError}
        error={sendError}
        onmodel={changeModel}
        onlanguage={(next) => (language = next)}
        onsend={send}
        onembodiment={selectEmbodiment}
        onrefresh={() => void refreshEmbodiments()}
        onopenbody={(id) => void openEmbodiment(id)}
      />
    {/if}
  </div>
{/if}
