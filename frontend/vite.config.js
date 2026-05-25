// frontend/vite.config.js
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const port = Number(env.VITE_PORT) || 3000; 

  return {
    plugins: [react()],
    server: {
      port: port,
      open: true,
      // 💡 Removed the /api proxy rewrite completely!
    },
    optimizeDeps: {
      include: ['react', 'react-dom', 'd3-scale', 'd3-interpolate']
    }
  };
});