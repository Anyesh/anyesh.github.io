import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import artifacts from "./vite-artifacts-plugin.js";

export default defineConfig({
  base: "/",
  plugins: [react(), artifacts()],
});
