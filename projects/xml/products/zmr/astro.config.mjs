import { defineConfig } from 'astro/config';
import tailwind from "@astrojs/tailwind";
import mdx from "@astrojs/mdx";
import node from "@astrojs/node";

import sitemap from "@astrojs/sitemap";

// https://astro.build/config
export default defineConfig({
  site: "https://blog.misile.xyz",
  integrations: [tailwind(), mdx({
    shikiConfig: {
      theme: "catppuccin-mocha",
      wrap: true
    }
  }), sitemap({
    filter: (p)=>p !== "https://blog.misile.xyz/" && !p.startsWith("https://blog.misile.xyz/pages/")
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
