import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Proxies the Vite dev server to the FastAPI voice backend so the
// browser can use same-origin ws://localhost:5173/ws/voice.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/ws': {
        target: 'ws://127.0.0.1:8000',
        ws: true,
        changeOrigin: true,
      },
      '/health': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
