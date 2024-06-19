// vite.config.ts
import { defineConfig } from "file:///home/misile/repos/h3/projects/dsb/ppts/ana/fastapi6/node_modules/.pnpm/vite@5.3.1_@types+node@20.14.5/node_modules/vite/dist/node/index.js";
import { svelte } from "file:///home/misile/repos/h3/projects/dsb/ppts/ana/fastapi6/node_modules/.pnpm/@sveltejs+vite-plugin-svelte@3.1.1_svelte@4.2.18_vite@5.3.1/node_modules/@sveltejs/vite-plugin-svelte/src/index.js";
import path from "path";
var __vite_injected_original_dirname = "/home/misile/repos/h3/projects/dsb/ppts/ana/fastapi6";
var vite_config_default = defineConfig({
  plugins: [svelte()],
  resolve: {
    alias: {
      "@config": path.resolve(__vite_injected_original_dirname, "./src/config.ts"),
      "@components": path.resolve(__vite_injected_original_dirname, "./src/lib/components/index.ts"),
      "@motion": path.resolve(__vite_injected_original_dirname, "./src/lib/motion/index.ts"),
      "@languages": path.resolve(__vite_injected_original_dirname, "./src/lib/languages/index.ts"),
      "@lib": path.resolve(__vite_injected_original_dirname, "./src/lib"),
      "@stores": path.resolve(__vite_injected_original_dirname, "./src/lib/stores"),
      "@styles": path.resolve(__vite_injected_original_dirname, "./src/lib/styles")
    }
  },
  base: "/fastapi6"
});
export {
  vite_config_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsidml0ZS5jb25maWcudHMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImNvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9kaXJuYW1lID0gXCIvaG9tZS9taXNpbGUvcmVwb3MvaDMvcHJvamVjdHMvZHNiL3BwdHMvYW5hL2Zhc3RhcGk2XCI7Y29uc3QgX192aXRlX2luamVjdGVkX29yaWdpbmFsX2ZpbGVuYW1lID0gXCIvaG9tZS9taXNpbGUvcmVwb3MvaDMvcHJvamVjdHMvZHNiL3BwdHMvYW5hL2Zhc3RhcGk2L3ZpdGUuY29uZmlnLnRzXCI7Y29uc3QgX192aXRlX2luamVjdGVkX29yaWdpbmFsX2ltcG9ydF9tZXRhX3VybCA9IFwiZmlsZTovLy9ob21lL21pc2lsZS9yZXBvcy9oMy9wcm9qZWN0cy9kc2IvcHB0cy9hbmEvZmFzdGFwaTYvdml0ZS5jb25maWcudHNcIjtpbXBvcnQgeyBkZWZpbmVDb25maWcgfSBmcm9tICd2aXRlJ1xuaW1wb3J0IHsgc3ZlbHRlIH0gZnJvbSAnQHN2ZWx0ZWpzL3ZpdGUtcGx1Z2luLXN2ZWx0ZSdcbmltcG9ydCBwYXRoIGZyb20gJ3BhdGgnXG5cbi8vIGh0dHBzOi8vdml0ZWpzLmRldi9jb25maWcvXG5leHBvcnQgZGVmYXVsdCBkZWZpbmVDb25maWcoe1xuXHRwbHVnaW5zOiBbc3ZlbHRlKCldLFxuXHRyZXNvbHZlOiB7XG5cdFx0YWxpYXM6IHtcblx0XHRcdCdAY29uZmlnJzogcGF0aC5yZXNvbHZlKF9fZGlybmFtZSwgJy4vc3JjL2NvbmZpZy50cycpLFxuXHRcdFx0J0Bjb21wb25lbnRzJzogcGF0aC5yZXNvbHZlKF9fZGlybmFtZSwgJy4vc3JjL2xpYi9jb21wb25lbnRzL2luZGV4LnRzJyksXG5cdFx0XHQnQG1vdGlvbic6IHBhdGgucmVzb2x2ZShfX2Rpcm5hbWUsICcuL3NyYy9saWIvbW90aW9uL2luZGV4LnRzJyksXG5cdFx0XHQnQGxhbmd1YWdlcyc6IHBhdGgucmVzb2x2ZShfX2Rpcm5hbWUsICcuL3NyYy9saWIvbGFuZ3VhZ2VzL2luZGV4LnRzJyksXG5cdFx0XHQnQGxpYic6IHBhdGgucmVzb2x2ZShfX2Rpcm5hbWUsICcuL3NyYy9saWInKSxcblx0XHRcdCdAc3RvcmVzJzogcGF0aC5yZXNvbHZlKF9fZGlybmFtZSwgJy4vc3JjL2xpYi9zdG9yZXMnKSxcblx0XHRcdCdAc3R5bGVzJzogcGF0aC5yZXNvbHZlKF9fZGlybmFtZSwgJy4vc3JjL2xpYi9zdHlsZXMnKSxcblx0XHR9LFxuXHR9LFxuXHRiYXNlOiBcIi9mYXN0YXBpNlwiXG59KVxuIl0sCiAgIm1hcHBpbmdzIjogIjtBQUE4VSxTQUFTLG9CQUFvQjtBQUMzVyxTQUFTLGNBQWM7QUFDdkIsT0FBTyxVQUFVO0FBRmpCLElBQU0sbUNBQW1DO0FBS3pDLElBQU8sc0JBQVEsYUFBYTtBQUFBLEVBQzNCLFNBQVMsQ0FBQyxPQUFPLENBQUM7QUFBQSxFQUNsQixTQUFTO0FBQUEsSUFDUixPQUFPO0FBQUEsTUFDTixXQUFXLEtBQUssUUFBUSxrQ0FBVyxpQkFBaUI7QUFBQSxNQUNwRCxlQUFlLEtBQUssUUFBUSxrQ0FBVywrQkFBK0I7QUFBQSxNQUN0RSxXQUFXLEtBQUssUUFBUSxrQ0FBVywyQkFBMkI7QUFBQSxNQUM5RCxjQUFjLEtBQUssUUFBUSxrQ0FBVyw4QkFBOEI7QUFBQSxNQUNwRSxRQUFRLEtBQUssUUFBUSxrQ0FBVyxXQUFXO0FBQUEsTUFDM0MsV0FBVyxLQUFLLFFBQVEsa0NBQVcsa0JBQWtCO0FBQUEsTUFDckQsV0FBVyxLQUFLLFFBQVEsa0NBQVcsa0JBQWtCO0FBQUEsSUFDdEQ7QUFBQSxFQUNEO0FBQUEsRUFDQSxNQUFNO0FBQ1AsQ0FBQzsiLAogICJuYW1lcyI6IFtdCn0K
