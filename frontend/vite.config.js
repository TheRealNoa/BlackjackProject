import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  define: {
    // amazon-cognito-identity-js → buffer expects Node's `global` in the browser bundle
    global: "globalThis",
  },
});
