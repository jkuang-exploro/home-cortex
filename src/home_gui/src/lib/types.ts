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
  agent_entity_id?: string | null;
  active_embodiment_id: string | null;
  language: string;
  greeting: string | null;
  title: string;
  updated_at: string;
}

export interface EmbodimentSummary {
  id: string;
  name: string;
  agent: { id: string; name: string | null } | null;
  linked: boolean;
  connected: boolean;
  state: 'unlinked' | 'linked_offline' | 'linked_online';
  runtime: { connected_at: string | null; last_seen: string | null };
  telemetry: {
    available: boolean;
    valid: boolean;
    fresh: boolean;
    space_id: string | null;
    measured_at: string | null;
  };
  capabilities: { name: string; supported: boolean; available: boolean }[];
}

export interface EmbodimentDetail extends EmbodimentSummary {
  geometry: { box: {
    length_m: number;
    width_m: number;
    height_m: number;
    center: { x: number; y: number; z: number };
  } } | null;
  local_frame: { forward: string; left: string; up: string } | null;
  telemetry: EmbodimentSummary['telemetry'] & {
    estimate: null | {
      embodiment_id: string;
      space_id: string | null;
      measured_at: string;
      validity: 'valid' | 'no_estimate';
      transform: null | Record<'x' | 'y' | 'z' | 'yaw' | 'pitch' | 'roll',
        { value: number; p95: number }>;
    };
  };
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
