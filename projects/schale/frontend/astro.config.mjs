import { defineConfig } from 'astro/config';

import tailwind from "@astrojs/tailwind";

import node from '@astrojs/node';
import sitemap from '@astrojs/sitemap';

// https://astro.build/config
export default defineConfig({
  site: "https://misile.xyz",
  integrations: [tailwind(), sitemap({
    filter: (route) => {
      const vs = ["/news"]
      for (const v of vs) {
        if (route.startsWith(v)) {
          return false;
        }
      }
      return true;
    }
  })],
  output: 'server',
  adapter: node({
    mode: 'standalone'
  })
});