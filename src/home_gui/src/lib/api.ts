import { ApiError, type Conversation, type Model, type Session } from './types';

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }
  const response = await fetch(path, { ...init, headers, credentials: 'include' });
  if (response.status === 204) {
    return undefined as T;
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = payload.error ?? {};
    throw new ApiError(
      response.status,
      error.code || `http_${response.status}`,
      error.message || response.statusText,
    );
  }
  return payload as T;
}

export function readSession(): Promise<Session> {
  return request('/session');
}

export function createSession(email: string, apiKey: string): Promise<Session> {
  return request('/session', {
    method: 'POST',
    headers: { Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify({ email }),
  });
}

export function deleteSession(): Promise<void> {
  return request('/session', { method: 'DELETE' });
}

export async function listModels(): Promise<Model[]> {
  const payload = await request<{ data: Model[] }>('/v1/models');
  return payload.data ?? [];
}

export async function listConversations(): Promise<Conversation[]> {
  const payload = await request<{ data: Conversation[] }>('/conversations');
  return payload.data ?? [];
}

export function createConversation(model: string, language: string): Promise<Conversation> {
  return request('/conversations', {
    method: 'POST',
    body: JSON.stringify({ model, language }),
  });
}

export function getConversation(id: string): Promise<Conversation> {
  return request(`/conversations/${id}`);
}

export function deleteConversation(id: string): Promise<void> {
  return request(`/conversations/${id}`, { method: 'DELETE' });
}

export async function* streamMessage(
  conversationId: string,
  content: string,
): AsyncGenerator<string> {
  const response = await fetch(`/conversations/${conversationId}/messages`, {
    method: 'POST',
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify({ content, stream: true }),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const error = payload.error ?? {};
    throw new ApiError(
      response.status,
      error.code || `http_${response.status}`,
      error.message || response.statusText,
    );
  }
  if (!response.body) {
    throw new ApiError(502, 'stream_unavailable', 'The browser did not expose the reply stream');
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split('\n\n');
    buffer = parts.pop() ?? '';
    for (const part of parts) {
      const line = part.split('\n').find((item) => item.startsWith('data: '));
      if (!line) continue;
      const data = line.slice(6).trim();
      if (!data || data === '[DONE]') {
        if (data === '[DONE]') return;
        continue;
      }
      let event: { error?: { code?: string; message?: string }; choices?: { delta?: { content?: string } }[] };
      try {
        event = JSON.parse(data);
      } catch {
        continue;
      }
      if (event.error) {
        throw new ApiError(502, event.error.code || 'stream_error', event.error.message || 'Stream failed');
      }
      const delta = event.choices?.[0]?.delta?.content;
      if (delta) yield delta;
    }
  }
}
