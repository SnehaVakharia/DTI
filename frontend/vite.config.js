import { defineConfig, loadEnv } from "vite";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const proxyTarget = env.VITE_PROXY_TARGET || "http://localhost:8000";

  return {
    server: {
      port: 3000,
      host: true,
      proxy: {
        "/api": proxyTarget,
      },
    },
  };
});
