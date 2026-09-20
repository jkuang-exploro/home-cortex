export type MediaType = 'photo' | 'video';
export type MediaFilter = 'all' | MediaType;

export interface MediaItem {
  id: string;
  type: MediaType;
  filename: string;
  relative_path: string;
  file_size: number;
  captured_at: string;
  width: number | null;
  height: number | null;
  duration_seconds: number | null;
  orientation: number | null;
  camera_make: string | null;
  camera_model: string | null;
  video_codec: string | null;
  metadata_status: string;
  thumbnail_status: string;
  thumbnail_url: string;
  content_url: string;
}

export interface MediaPage {
  items: MediaItem[];
  total: number;
  limit: number;
  offset: number;
  next_offset: number | null;
}

async function mediaRequest<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`/media-api${path}`, { ...init, credentials: 'include' });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = payload.error ?? {};
    const failure = new Error(error.message || response.statusText || 'Media service unavailable');
    Object.assign(failure, { status: response.status, code: error.code });
    throw failure;
  }
  return payload as T;
}

export function listMedia(type: MediaFilter, offset = 0, limit = 100): Promise<MediaPage> {
  const query = new URLSearchParams({ type, offset: String(offset), limit: String(limit) });
  return mediaRequest<MediaPage>(`/items?${query}`);
}

export function refreshMedia(): Promise<Record<string, unknown>> {
  return mediaRequest('/refresh', { method: 'POST' });
}

