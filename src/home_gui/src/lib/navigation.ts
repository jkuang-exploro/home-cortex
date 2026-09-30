export type Page = 'chat' | 'media' | 'embodiments';

export interface Route {
  page: Page;
  embodimentId: string | null;
}

export function routeFromPath(pathname: string): Route {
  if (pathname === '/media') return { page: 'media', embodimentId: null };
  if (pathname === '/embodiments' || pathname === '/embodiments/') {
    return { page: 'embodiments', embodimentId: null };
  }
  const match = /^\/embodiments\/([^/]+)\/?$/.exec(pathname);
  if (match) {
    try {
      const id = decodeURIComponent(match[1]);
      if (id) return { page: 'embodiments', embodimentId: id };
    } catch {
      // Malformed paths fall back to the ordinary chat route.
    }
  }
  return { page: 'chat', embodimentId: null };
}

export function pathForRoute(route: Route): string {
  if (route.page === 'media') return '/media';
  if (route.page === 'embodiments') {
    return route.embodimentId
      ? `/embodiments/${encodeURIComponent(route.embodimentId)}`
      : '/embodiments';
  }
  return '/';
}
