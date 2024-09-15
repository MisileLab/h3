import { defineConfig } from 'astro/config';

import tailwind from "@astrojs/tailwind";

import node from '@astrojs/node';
import sitemap from '@astrojs/sitemap';

import mdx from '@astrojs/mdx';

// https://astro.build/config
export default defineConfig({
  integrations: [tailwind(), sitemap({
    filter: ({ route }) => {
      const vs = ["/news"]
      for (const v of vs) {
        if (route.startsWith(v)) {
          return false;
        }
      }
    }
  }), mdx()],
  output: 'server',
  adapter: node({
    mode: 'standalone'
  })
});