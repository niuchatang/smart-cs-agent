import { createApp } from 'vue'
import './style.css'
import 'leaflet/dist/leaflet.css'
import App from './App.vue'
import router from './router'

try {
  document.body.classList.toggle('theme-day', localStorage.getItem('ui-theme') === 'day')
} catch (_) {
  /* ignore */
}

createApp(App).use(router).mount('#app')
