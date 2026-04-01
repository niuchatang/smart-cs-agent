import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  base: '/static-vue/',
  plugins: [vue()],
  build: {
    outDir: '../static-vue',
    emptyOutDir: true,
  },
})
