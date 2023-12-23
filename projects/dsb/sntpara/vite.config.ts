import { defineConfig } from "vite";
import solidPlugin from "vite-plugin-solid";
import devtools from 'solid-devtools/vite';
import compression from 'vite-plugin-compression2';

export default defineConfig({
  plugins: [
    /* 
    Uncomment the following line to enable solid-devtools.
    For more info see https://github.com/thetarnav/solid-devtools/tree/main/packages/extension#readme
    */
    devtools(),
    solidPlugin(),
    compression()
  ],
  server: {
    port: 3000,
  },
  build: {
    target: "esnext",
  },
});
