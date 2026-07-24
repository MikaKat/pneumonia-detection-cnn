import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// During local dev the UI runs on :5173 and your Python API on :8000.
// This proxy forwards /api -> the backend so the browser sees one origin
// (no CORS needed). Override the target with VITE_DEV_API_TARGET if your
// backend listens elsewhere.
export default defineConfig(({ mode }) => {
  // Use 127.0.0.1 (IPv4), NOT "localhost": Node resolves localhost to IPv6 (::1)
  // first, but uvicorn binds to IPv4 -> proxy would fail with ECONNREFUSED ::1:8000.
  const devTarget = process.env.VITE_DEV_API_TARGET || "http://127.0.0.1:8000";
  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        "/api": {
          target: devTarget,
          changeOrigin: true,
        },
      },
    },
    build: {
      outDir: "dist",
      sourcemap: false,
    },
  };
});
