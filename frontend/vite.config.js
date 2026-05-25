// frontend/vite.config.js
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  // Load env file from the current directory (frontend/)
  // process.cwd() tells Vite to look right here for the .env file
  const env = loadEnv(mode, process.cwd(), '');

  // Fallback to 3000 
  const port = Number(env.VITE_PORT) || 3000; 

  return {
    plugins: [react()],
    server: {
      port: port,
      open: true,
      // Proxy API requests to the backend server
      proxy: {
        '/api': {
          target: 'http://localhost:5000', //  Backend URL
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, ''),
        },
      },
    },
    optimizeDeps: {
      include: ['react', 'react-dom', 'd3-scale', 'd3-interpolate']
    }
  };
});