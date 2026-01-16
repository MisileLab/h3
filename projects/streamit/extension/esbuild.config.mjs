import * as esbuild from 'esbuild';

const isWatch = process.argv.includes('--watch');

/** @type {esbuild.BuildOptions} */
const commonOptions = {
  bundle: true,
  sourcemap: true,
  target: 'es2022',
  format: 'esm',
  outdir: 'dist',
  logLevel: 'info',
};

// Main extension scripts
const mainBuild = {
  ...commonOptions,
  entryPoints: [
    'src/content.ts',
    'src/popup.ts',
    'src/sw.ts',
    'src/offscreen.ts',
  ],
};

// AudioWorklet processor (needs special handling - no bundling of externals)
const workletBuild = {
  ...commonOptions,
  entryPoints: ['src/worklet-processor.ts'],
  // Worklets run in isolated context, keep it simple
  format: 'iife',
  globalName: undefined,
};

async function build() {
  try {
    if (isWatch) {
      const mainCtx = await esbuild.context(mainBuild);
      const workletCtx = await esbuild.context(workletBuild);
      
      await Promise.all([
        mainCtx.watch(),
        workletCtx.watch(),
      ]);
      
      console.log('Watching for changes...');
    } else {
      await Promise.all([
        esbuild.build(mainBuild),
        esbuild.build(workletBuild),
      ]);
      
      console.log('Build complete!');
    }
  } catch (error) {
    console.error('Build failed:', error);
    process.exit(1);
  }
}

build();
