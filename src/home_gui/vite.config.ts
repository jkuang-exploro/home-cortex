import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

const api = 'http://127.0.0.1:8001';

export default defineConfig({
  plugins: [svelte()],
  server: {
    port: 5173,
    proxy: {
      '/session': api,
      '/conversations': api,
      '/agent': api,
      '/v1': api,
      '/health': api,
    },
  },
  preview: { port: 4173, host: true },
});
