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

type MediaApiError = Error & { status?: number; code?: string };

function mediaError(message: string, status?: number, code?: string): MediaApiError {
  return Object.assign(new Error(message), { status, code });
}

async function mediaRequest(
  path: string,
  init: RequestInit = {},
  timeoutMs = 15_000,
): Promise<unknown> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  let response: Response;
  try {
    response = await fetch(`/media-api${path}`, {
      ...init,
      credentials: 'include',
      signal: controller.signal,
    });
  } catch (failure) {
    if (failure instanceof DOMException && failure.name === 'AbortError') {
      const seconds = Math.round(timeoutMs / 1000);
      throw mediaError(`The media service did not respond within ${seconds} seconds.`, 504, 'media_timeout');
    }
    throw failure;
  } finally {
    window.clearTimeout(timeout);
  }

  const contentType = response.headers.get('content-type') ?? '';
  if (!contentType.toLowerCase().includes('application/json')) {
    if (!response.ok) {
      throw mediaError(
        `Media service request failed (${response.status}).`,
        response.status,
        `http_${response.status}`,
      );
    }
    throw mediaError(
      'The media API returned the web app instead of JSON. Rebuild and restart nginx and home-media together.',
      502,
      'invalid_media_response',
    );
  }

  const payload = await response.json().catch(() => {
    throw mediaError('The media API returned malformed JSON.', 502, 'invalid_media_response');
  });
  if (!response.ok) {
    const envelope = typeof payload === 'object' && payload !== null
      ? payload as { error?: { message?: string; code?: string } }
      : {};
    const error = envelope.error ?? {};
    throw mediaError(
      error.message || response.statusText || 'Media service unavailable',
      response.status,
      error.code || `http_${response.status}`,
    );
  }
  return payload;
}

export async function listMedia(type: MediaFilter, offset = 0, limit = 100): Promise<MediaPage> {
  const query = new URLSearchParams({ type, offset: String(offset), limit: String(limit) });
  const payload = await mediaRequest(`/items?${query}`);
  if (!isMediaPage(payload)) {
    throw mediaError(
      'The media API returned an invalid item list. Rebuild and restart nginx and home-media together.',
      502,
      'invalid_media_response',
    );
  }
  return payload;
}

export async function refreshMedia(): Promise<Record<string, unknown>> {
  const payload = await mediaRequest('/refresh', { method: 'POST' }, 600_000);
  if (typeof payload !== 'object' || payload === null || Array.isArray(payload)) {
    throw mediaError('The media refresh returned an invalid response.', 502, 'invalid_media_response');
  }
  return payload as Record<string, unknown>;
}

function isMediaPage(value: unknown): value is MediaPage {
  if (typeof value !== 'object' || value === null) return false;
  const page = value as Partial<MediaPage>;
  return Array.isArray(page.items)
    && typeof page.total === 'number'
    && typeof page.limit === 'number'
    && typeof page.offset === 'number'
    && (page.next_offset === null || typeof page.next_offset === 'number');
}
