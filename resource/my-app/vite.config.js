import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'

// Vite config for Dataiku DSS plugin webapp integration
// - base: ensure built asset URLs resolve from the webapp context to resources
// - build.outDir: output to `build/` so it matches plugin docs and symlink target
export default defineConfig({
  plugins: [react()],
  // Use relative base; we'll copy the entire build into webapps/react
  base: './',
  build: {
    outDir: 'build',
    emptyOutDir: true,
  },
})
