import { defineConfig } from 'astro/config';
import tailwind from "@astrojs/tailwind";
import mdx from "@astrojs/mdx";

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
    filter: (p)=>p !== "https://blog.misile.xyz/" && !p.startsWith("https://blog.misile.xyz/pages/") && !p.startsWith("https://blog.misile.xyz/news/")
  })],
  markdown: {
    shikiConfig: {
      theme: "catppuccin-mocha",
      wrap: true
    }
  },
  experimental: {
    contentCollectionCache: true
  }
});
