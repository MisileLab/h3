import { defineConfig } from 'astro/config';
import tailwind from "@astrojs/tailwind";

import mdx from "@astrojs/mdx";

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
  }
});