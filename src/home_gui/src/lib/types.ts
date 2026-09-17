export type Language = 'en' | 'zh';

export type ModelKind = 'agent' | 'model';

export interface Model {
  id: string;
  owned_by: string;
  kind?: ModelKind;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  created_at?: string;
}

export interface ConversationSummary {
  id: string;
  model: string;
  agent_id: string | null;
  language: string;
  greeting: string | null;
  title: string;
  updated_at: string;
}

export interface Conversation extends ConversationSummary {
  messages: ChatMessage[];
}

export interface Session {
  object: 'session';
  email?: string;
  user_id?: string;
  anonymous?: boolean;
}

export class ApiError extends Error {
  status: number;
  code: string;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}
