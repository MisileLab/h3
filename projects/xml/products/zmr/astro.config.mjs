import { defineConfig } from 'astro/config';
import tailwind from "@astrojs/tailwind";
import mdx from "@astrojs/mdx";

import node from "@astrojs/node";

// https://astro.build/config
export default defineConfig({
  integrations: [tailwind(), mdx({
    shikiConfig: {
      theme: "catppuccin-mocha",
      wrap: true
    }
  })],
  markdown: {
    shikiConfig: {
      theme: "catppuccin-mocha",
      wrap: true
    }
  },
  redirects: {
    "/": "/pages/1",
    "/pages": "/pages/1",
    "/posts": "/pages/1"
  },
  output: "server",
  adapter: node({
    mode: "standalone"
  })
});