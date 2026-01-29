import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig(({ mode }) => {
    const envFile = loadEnv(mode, process.cwd(), "");
    const env = { ...process.env, ...envFile };

    return {
        plugins: [tailwindcss(), react()],
        server: {
            host: true,
            port: parseInt(env.FRONTEND_PORT) || 3030,
            proxy: {
                "/api": {
                    target:
                        env.BACKEND_URL || `http://backend:${env.BACKEND_PORT}`,
                    changeOrigin: true,
                },
            },
        },
    };
});
