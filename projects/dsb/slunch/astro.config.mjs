import { defineConfig } from 'astro/config';
import tailwind from "@astrojs/tailwind";

import vitePwa from "@vite-pwa/astro";

// https://astro.build/config
export default defineConfig({
  integrations: [tailwind(), vitePwa()]
});