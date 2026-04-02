import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 开发时把 API 转到本机 FastAPI（默认 8010），否则 fetch('/auth/...') 会打到 Vite 自身导致注册/登录无效
const API_TARGET = process.env.VITE_API_TARGET || 'http://127.0.0.1:8010'

export default defineConfig({
  base: '/static-vue/',
  plugins: [vue()],
  build: {
    outDir: '../static-vue',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/auth': { target: API_TARGET, changeOrigin: true },
      '/chat': { target: API_TARGET, changeOrigin: true },
      '/history': { target: API_TARGET, changeOrigin: true },
      '/health': { target: API_TARGET, changeOrigin: true },
      '/debug': { target: API_TARGET, changeOrigin: true },
    },
  },
})
