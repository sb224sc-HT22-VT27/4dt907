/* global process */
import path from "node:path";
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig(({ mode }) => {
  // Vite is started from src/frontend, but your shared .env is in src/.
  const envDir = path.resolve(process.cwd(), "..");
  const env = loadEnv(mode, envDir, "");

  // we can use it if ports collide @N.A
  const mergedEnv = { ...env, ...process.env };

  const frontendPort = Number(mergedEnv.FRONTEND_PORT) || 3030;

  // Dynamic backend target:
  // if BACKEND_URL is provided, use it
  // else use localhost with BACKEND_PORT
  // This makes local dev work by default.
  const backendPort = Number(mergedEnv.BACKEND_PORT) || 8080;
  const backendUrl = mergedEnv.BACKEND_URL || `http://localhost:${backendPort}`;

  return {
    plugins: [tailwindcss(), react()],
    server: {
      host: true,
      port: frontendPort,
      proxy: {
        "/api": {
          target: backendUrl,
          changeOrigin: true,
          secure: false
        }
      }
    }
  };
});
