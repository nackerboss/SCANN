import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 2000, // Set your desired port number
    host: '0.0.0.0',
  },
});