<script setup>
import L from 'leaflet'
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { apiDelete, apiGet, apiPost } from '../api'
import { buildRouteSegments, enrichHighway } from '../utils/routeDisplay.js'

const router = useRouter()
const CONV_KEY = 'chat-conversation-id'
const WELCOME_TEXT = '你好，我是智慧交通客服助手。你可以问我：路径规划、线路状态、高速事故/管制、票价规则、退票、失物招领、投诉等问题。'

const currentUsername = ref('')
const currentConversationId = ref(localStorage.getItem(CONV_KEY) || 'default')
const messages = ref([])
const inputMessage = ref('')
const sending = ref(false)
const isDayTheme = ref(localStorage.getItem('ui-theme') === 'day')
const routeResult = ref(null)
const layoutRowRef = ref(null)
const routeCardRef = ref(null)
const layoutLockedPx = ref(null)
const chatBoxRef = ref(null)
const mapRef = ref(null)
const highwaySceneRef = ref(null)
const carRef = ref(null)
const trafficLayerRef = ref(null)
let map = null
let routeLayer = null
let markerLayer = null
let congestionLayer = null
let congestionRouteLayer = null
let layoutResizeDebounceTimer = null
let layoutResizeObserver = null
let routeCardResizeObserver = null
let routeCardResizeTimer = null
let onLayoutWindowResize = null

function teardownRouteCardResizeObserver() {
  if (routeCardResizeObserver) {
    routeCardResizeObserver.disconnect()
    routeCardResizeObserver = null
  }
  clearTimeout(routeCardResizeTimer)
}

function setupRouteCardResizeObserver() {
  teardownRouteCardResizeObserver()
  const card = routeCardRef.value
  if (!card || !routeResult.value || typeof ResizeObserver === 'undefined') return
  routeCardResizeObserver = new ResizeObserver(() => {
    if (!layoutRowRef.value?.classList.contains('layout--height-locked')) return
    clearTimeout(routeCardResizeTimer)
    routeCardResizeTimer = setTimeout(() => scheduleFreezeLayoutHeight(), 160)
  })
  routeCardResizeObserver.observe(card)
}

const routeBadge = computed(() => {
  if (!routeResult.value) return '等待查询'
  const sourceMap = {
    amap_direction: '高德路径 API',
    osrm: '实时路径 API',
    fallback_estimation: '本地估算路径',
  }
  return sourceMap[routeResult.value.source] || '路径结果'
})
const routeDistanceText = computed(() => (routeResult.value ? `${routeResult.value.distance_km ?? '--'} km` : '-- km'))
const routeDurationText = computed(() => (routeResult.value ? `${routeResult.value.duration_min ?? '--'} 分钟` : '-- 分钟'))
const routeOriginText = computed(() => routeResult.value?.origin || '请在右侧输入“从A到B怎么走”')
const routeDestinationText = computed(() => routeResult.value?.destination || '等待规划结果')
const routeSegments = computed(() => buildRouteSegments(routeResult.value))
const tripHints = computed(() => {
  const th = routeResult.value?.trip_hints
  if (!Array.isArray(th)) return []
  return th.filter((x) => x && String(x.title || '').trim() && String(x.detail || '').trim())
})

function unfreezeLayoutHeight() {
  const el = layoutRowRef.value
  if (!el) return
  el.classList.remove('layout--height-locked')
  el.style.height = ''
  el.style.minHeight = ''
  layoutLockedPx.value = null
}

function clampLayoutIfLocked() {
  const el = layoutRowRef.value
  const cap = layoutLockedPx.value
  if (!el || cap == null || !el.classList.contains('layout--height-locked')) return
  const now = Math.ceil(el.getBoundingClientRect().height)
  if (now > cap + 1) {
    el.style.height = `${cap}px`
    el.style.minHeight = `${cap}px`
  }
}

/** 按左侧路径卡片自然高度锁定整行；右侧对话区同高，仅在 .chat-box 内滚动 */
function freezeLayoutHeightNow() {
  const layout = layoutRowRef.value
  const routeCard = routeCardRef.value
  if (!layout || !routeCard || !routeResult.value) return
  const pts = routeResult.value.route_points
  const hasPath = Array.isArray(pts) && pts.length >= 2
  if (!hasPath) return
  const th = routeResult.value.trip_hints
  const hintsReady = !Array.isArray(th) || th.length === 0 || tripHints.value.length > 0
  if (!hintsReady) return
  if (map) map.invalidateSize()
  const hRaw = Math.ceil(routeCard.getBoundingClientRect().height)
  const h = Math.max(280, hRaw)
  layoutLockedPx.value = h
  layout.style.height = `${h}px`
  layout.style.minHeight = `${h}px`
  layout.classList.add('layout--height-locked')
  if (map) map.invalidateSize()
}

function scheduleFreezeLayoutHeight() {
  unfreezeLayoutHeight()
  nextTick(() => {
    nextTick(() => {
      setTimeout(() => {
        if (map) map.invalidateSize()
        requestAnimationFrame(() => {
          freezeLayoutHeightNow()
          setTimeout(() => {
            if (map) map.invalidateSize()
          }, 220)
        })
      }, 280)
    })
  })
}

function scrollChatToBottom() {
  nextTick(() => {
    if (chatBoxRef.value) chatBoxRef.value.scrollTop = chatBoxRef.value.scrollHeight
    clampLayoutIfLocked()
  })
}

function normalizeFollowUps(raw) {
  if (!Array.isArray(raw) || !raw.length) return undefined
  const out = raw
    .map((x) => ({
      question: String(x?.question ?? '').trim(),
      answer: String(x?.answer ?? '').trim(),
    }))
    .filter((x) => x.question.length >= 2)
  return out.length ? out.slice(0, 3) : undefined
}

function appendMessage(role, text, followUps) {
  const fu = normalizeFollowUps(followUps)
  messages.value.push({
    role,
    text,
    followUps: fu,
  })
  scrollChatToBottom()
  nextTick(() => clampLayoutIfLocked())
}

function applyTheme() {
  document.body.classList.toggle('theme-day', isDayTheme.value)
  localStorage.setItem('ui-theme', isDayTheme.value ? 'day' : 'night')
}

function toggleTheme() {
  isDayTheme.value = !isDayTheme.value
  applyTheme()
}

async function ensureAuth() {
  try {
    const me = await apiGet('/auth/me')
    currentUsername.value = me?.user?.username || ''
  } catch (_) {
    router.replace('/login')
    throw new Error('unauthorized')
  }
}

function initMap() {
  if (!mapRef.value || map) return
  // scrollWheelZoom：允许鼠标滚轮缩放；与页面纵向滚动同用时，仅在指针悬停地图时生效
  map = L.map(mapRef.value, { zoomControl: true, scrollWheelZoom: true }).setView([35.8617, 104.1954], 4)
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap',
  }).addTo(map)
  routeLayer = L.layerGroup().addTo(map)
  markerLayer = L.layerGroup().addTo(map)
  congestionLayer = L.layerGroup().addTo(map)
  congestionRouteLayer = L.layerGroup().addTo(map)
}

function renderRouteOnMap() {
  if (!map || !routeResult.value) return
  routeLayer.clearLayers()
  markerLayer.clearLayers()
  const points = Array.isArray(routeResult.value.route_points) ? routeResult.value.route_points : []
  if (points.length < 2) return
  const polyline = L.polyline(points, { color: '#3b82f6', weight: 5, opacity: 0.9 }).addTo(routeLayer)
  const start = points[0]
  const end = points[points.length - 1]
  L.circleMarker(start, { radius: 6, color: '#10b981', fillColor: '#10b981', fillOpacity: 1 }).addTo(markerLayer).bindPopup(`起点：${routeResult.value.origin || '起点'}`)
  L.circleMarker(end, { radius: 6, color: '#ef4444', fillColor: '#ef4444', fillOpacity: 1 }).addTo(markerLayer).bindPopup(`终点：${routeResult.value.destination || '终点'}`)
  map.fitBounds(polyline.getBounds(), { padding: [24, 24] })
  nextTick(() => {
    if (map) map.invalidateSize()
  })
}

function getCongestionSeverity(level) {
  const t = String(level || '')
  // 必须先判轻度/中度，避免「轻度拥堵」里的「拥堵」被当成严重
  if (/(严重|重度|特别拥堵|瘫痪|堵死|红色|红\b)/.test(t)) return 'red'
  if (/(轻度|中度|缓行|慢行|拥挤|黄色|黄\b)/.test(t)) return 'yellow'
  if (/拥堵|堵塞/.test(t)) return 'yellow'
  return 'normal'
}

function haversineMeters(a, b) {
  const R = 6371000
  const toRad = (x) => (x * Math.PI) / 180
  const dLat = toRad(b[0] - a[0])
  const dLon = toRad(b[1] - a[1])
  const la1 = toRad(a[0])
  const la2 = toRad(b[0])
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(la1) * Math.cos(la2) * Math.sin(dLon / 2) ** 2
  return 2 * R * Math.asin(Math.min(1, Math.sqrt(h)))
}

function nearestRouteIndex(route, lat, lon) {
  let best = 0
  let bestD = Infinity
  for (let i = 0; i < route.length; i += 1) {
    const p = route[i]
    const d = haversineMeters([p[0], p[1]], [lat, lon])
    if (d < bestD) {
      bestD = d
      best = i
    }
  }
  return best
}

/** 以 centerIdx 为中心，沿路线向两侧各延伸约 halfKm 公里，只标「拥堵路段」 */
function extractRouteSubline(route, centerIdx, halfKm) {
  const halfM = halfKm * 1000
  let dist = 0
  let i = centerIdx
  while (i > 0 && dist < halfM) {
    dist += haversineMeters(route[i - 1], route[i])
    i -= 1
  }
  const start = i
  dist = 0
  i = centerIdx
  while (i < route.length - 1 && dist < halfM) {
    dist += haversineMeters(route[i], route[i + 1])
    i += 1
  }
  const end = i
  const slice = route.slice(start, end + 1)
  return slice.length >= 2 ? slice : null
}

function renderCongestionOverlay(conditionResults = []) {
  if (!map || !congestionLayer || !congestionRouteLayer) return
  congestionLayer.clearLayers()
  congestionRouteLayer.clearLayers()

  const successItems = (conditionResults || []).filter((x) => x && x.success)
  if (!successItems.length) return

  const route = routeResult.value && Array.isArray(routeResult.value.route_points) ? routeResult.value.route_points : null
  const halfKm = 14

  successItems.forEach((item) => {
    const severity = getCongestionSeverity(item.congestion_level)
    if (severity === 'normal') return
    const color = severity === 'red' ? '#ef4444' : '#facc15'
    const weight = severity === 'red' ? 10 : 7
    const opacity = severity === 'red' ? 0.88 : 0.82
    const label = `${item.name || item.code || '路段'}：${item.congestion_level || ''}`
    const probeList = Array.isArray(item.probe_points) ? item.probe_points : []

    if (route && route.length >= 2) {
      probeList.forEach((p) => {
        if (typeof p.lat !== 'number' || typeof p.lon !== 'number') return
        const idx = nearestRouteIndex(route, p.lat, p.lon)
        const sub = extractRouteSubline(route, idx, halfKm)
        if (sub) {
          L.polyline(sub, { color, weight, opacity, lineCap: 'round', lineJoin: 'round' })
            .addTo(congestionRouteLayer)
            .bindPopup(label)
        }
      })
    } else {
      probeList.forEach((p) => {
        if (typeof p.lat !== 'number' || typeof p.lon !== 'number') return
        L.circleMarker([p.lat, p.lon], {
          radius: severity === 'red' ? 10 : 8,
          color,
          fillColor: color,
          fillOpacity: 0.55,
          weight: 2,
        })
          .addTo(congestionLayer)
          .bindPopup(label)
      })
    }
  })
}

async function loadHistory() {
  messages.value = []
  try {
    const data = await apiGet(`/history/${encodeURIComponent(currentConversationId.value)}`)
    const items = data.items || []
    if (!items.length) {
      appendMessage('agent', WELCOME_TEXT)
      return
    }
    items.forEach((it) => {
      const role = it.role === 'agent' ? 'agent' : 'user'
      const fu = role === 'agent' ? normalizeFollowUps(it.meta?.follow_ups) : undefined
      messages.value.push({
        role,
        text: it.content,
        followUps: fu,
      })
    })
    scrollChatToBottom()
    nextTick(() => clampLayoutIfLocked())
  } catch (e) {
    appendMessage('agent', `加载历史失败: ${String(e)}`)
  }
}

async function postChatMessage(message) {
  sending.value = true
  try {
    const data = await apiPost('/chat', { user_id: currentConversationId.value, message })
    appendMessage('agent', data.reply, data.follow_ups)
    const rr = (data.tool_results || []).find((x) => x.tool === 'query_route_plan' && x.success)
    if (rr) {
      unfreezeLayoutHeight()
      routeResult.value = rr
      if (congestionLayer) congestionLayer.clearLayers()
      if (congestionRouteLayer) congestionRouteLayer.clearLayers()
      nextTick(() => scheduleFreezeLayoutHeight())
    }
    const conditionItems = (data.tool_results || []).filter((x) => x.tool === 'query_highway_condition' && x.success)
    if (conditionItems.length) {
      renderCongestionOverlay(conditionItems)
    }
  } catch (e) {
    if (String(e).includes('请先登录') || String(e).includes('登录状态已失效')) {
      router.replace('/login')
      return
    }
    appendMessage('agent', `请求失败: ${String(e)}`)
  } finally {
    sending.value = false
  }
}

async function sendMessage() {
  const message = inputMessage.value.trim()
  if (!message || sending.value) return
  appendMessage('user', message)
  inputMessage.value = ''
  await postChatMessage(message)
}

async function submitFollowUpQuestion(msgIdx, fi) {
  const m = messages.value[msgIdx]
  const q = m?.followUps?.[fi]?.question?.trim()
  if (!q || sending.value) return
  appendMessage('user', q)
  await postChatMessage(q)
}

async function clearChat() {
  try {
    await apiDelete(`/history/${encodeURIComponent(currentConversationId.value)}`)
  } catch (_) {}
  messages.value = [{ role: 'agent', text: WELCOME_TEXT }]
  if (congestionLayer) congestionLayer.clearLayers()
  if (congestionRouteLayer) congestionRouteLayer.clearLayers()
}

function newChat() {
  currentConversationId.value = `conv-${Date.now()}`
  localStorage.setItem(CONV_KEY, currentConversationId.value)
  messages.value = [{ role: 'agent', text: WELCOME_TEXT }]
  routeResult.value = null
  unfreezeLayoutHeight()
  if (congestionLayer) congestionLayer.clearLayers()
  if (congestionRouteLayer) congestionRouteLayer.clearLayers()
  scrollChatToBottom()
}

async function logout() {
  try {
    await apiPost('/auth/logout', {})
  } catch (_) {}
  router.replace('/login')
}

function onInputKeydown(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    sendMessage()
  }
}

function initTitleCarAnimation() {
  const highwayScene = highwaySceneRef.value
  const car = carRef.value
  const trafficLayer = trafficLayerRef.value
  if (!highwayScene || !car || !trafficLayer) return

  const carWidth = 66
  const sidePadding = 10
  const roadHeight = 80
  /* bottom 为距路面底部的 CSS 值；两车道的垂直区间需错开，避免与 22px 高的小车碰撞盒跨道重叠 */
  const lanes = [
    { bottom: 14, dir: 1 },
    { bottom: 54, dir: -1 },
  ]
  const minCarGap = 24
  const obstacles = []
  let carShiftX = 0
  let carBottom = lanes[0].bottom
  let lastTs = 0
  let spawnTimer = 0

  const renderCar = () => {
    car.style.left = '50%'
    car.style.bottom = `${carBottom.toFixed(2)}px`
    /* 横向只用 transform，避免每帧改 left 触发布局；子像素减少抖动 */
    car.style.transform = `translate3d(calc(-50% + ${carShiftX.toFixed(2)}px), 0, 0)`
  }

  const moveCar = (clientX, clientY) => {
    const rect = highwayScene.getBoundingClientRect()
    const ratioX = Math.min(1, Math.max(0, (clientX - rect.left) / Math.max(1, rect.width)))
    const dynamicMaxShift = Math.max(40, rect.width / 2 - carWidth / 2 - sidePadding)
    carShiftX = (ratioX - 0.5) * 2 * dynamicMaxShift
    const roadTopY = rect.height - roadHeight
    const mouseRoadY = Math.min(rect.height - 2, Math.max(roadTopY, clientY - rect.top))
    const laneSplit = roadTopY + roadHeight / 2
    carBottom = mouseRoadY < laneSplit ? lanes[1].bottom : lanes[0].bottom
    renderCar()
  }

  const spawnObstacle = () => {
    const el = document.createElement('div')
    el.className = 'obstacle-car'
    el.innerHTML = '<div class="obstacle-headlight top"></div><div class="obstacle-headlight bottom"></div><div class="obstacle-wheel left"></div><div class="obstacle-wheel right"></div>'
    const lane = lanes[Math.floor(Math.random() * lanes.length)]
    const laneBottom = lane.bottom
    const dir = lane.dir
    if (dir > 0) el.classList.add('reverse')
    const hasNearCar = obstacles.some((o) => {
      if (o.y !== laneBottom || o.dir !== dir) return false
      if (dir < 0) return o.x > trafficLayer.clientWidth - 230
      return o.x < 170
    })
    if (hasNearCar) return
    const startX = dir < 0 ? trafficLayer.clientWidth + 70 : -70
    el.style.bottom = `${laneBottom}px`
    trafficLayer.appendChild(el)
    obstacles.push({ el, x: startX, y: laneBottom, w: 58, h: 20, speed: 36, dir })
  }

  const intersects = (a, b) => a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y

  const resetRun = () => {
    for (const o of obstacles) o.el.remove()
    obstacles.length = 0
    carShiftX = 0
    carBottom = lanes[0].bottom
    spawnTimer = -0.8
    renderCar()
  }

  const tick = (ts) => {
    if (!lastTs) lastTs = ts
    const dt = Math.min(0.05, (ts - lastTs) / 1000)
    lastTs = ts
    spawnTimer += dt
    if (spawnTimer >= 2.1 && obstacles.length < 4) {
      spawnTimer = 0
      spawnObstacle()
    }
    const carRect = { x: trafficLayer.clientWidth / 2 + carShiftX - 33, y: carBottom, w: 66, h: 22 }
    for (let i = obstacles.length - 1; i >= 0; i -= 1) {
      const o = obstacles[i]
      o.x += o.dir * o.speed * dt
      if (o.x < -90 || o.x > trafficLayer.clientWidth + 90) {
        o.el.remove()
        obstacles.splice(i, 1)
      }
    }
    for (const lane of lanes) {
      const laneCars = obstacles.filter((o) => o.y === lane.bottom && o.dir === lane.dir).sort((a, b) => a.x - b.x)
      for (let i = 1; i < laneCars.length; i += 1) {
        if (lane.dir < 0) {
          const front = laneCars[i - 1]
          const back = laneCars[i]
          const minBackX = front.x + front.w + minCarGap
          if (back.x < minBackX) back.x = minBackX
        } else {
          const back = laneCars[i - 1]
          const front = laneCars[i]
          const maxBackX = front.x - back.w - minCarGap
          if (back.x > maxBackX) back.x = maxBackX
        }
      }
    }
    let crashed = false
    for (const o of obstacles) {
      o.el.style.transform = `translate3d(${o.x.toFixed(2)}px, 0, 0)`
      if (!crashed && intersects(carRect, o)) crashed = true
    }
    if (crashed) resetRun()
    requestAnimationFrame(tick)
  }

  highwayScene.addEventListener('mousemove', (e) => moveCar(e.clientX, e.clientY))
  highwayScene.addEventListener('mouseenter', (e) => moveCar(e.clientX, e.clientY))
  renderCar()
  requestAnimationFrame(tick)
}

watch(routeResult, (rr) => {
  nextTick(() => {
    teardownRouteCardResizeObserver()
    if (!map) return
    if (!rr || !Array.isArray(rr.route_points) || rr.route_points.length < 2) {
      if (routeLayer) routeLayer.clearLayers()
      if (markerLayer) markerLayer.clearLayers()
      return
    }
    renderRouteOnMap()
    nextTick(() => setupRouteCardResizeObserver())
  })
})

watch(tripHints, () => {
  if (!routeResult.value || !layoutRowRef.value) return
  scheduleFreezeLayoutHeight()
})

function highwayCardItems(list) {
  return (Array.isArray(list) ? list : []).map((raw) => ({ ...enrichHighway(raw), raw }))
}

onMounted(async () => {
  applyTheme()
  initTitleCarAnimation()
  initMap()
  onLayoutWindowResize = () => {
    const el = layoutRowRef.value
    if (!el || !el.classList.contains('layout--height-locked')) return
    clearTimeout(layoutResizeDebounceTimer)
    layoutResizeDebounceTimer = setTimeout(() => {
      unfreezeLayoutHeight()
      nextTick(() => {
        if (routeResult.value) scheduleFreezeLayoutHeight()
      })
    }, 200)
  }
  window.addEventListener('resize', onLayoutWindowResize)
  nextTick(() => {
    const el = layoutRowRef.value
    if (!el || typeof ResizeObserver === 'undefined') return
    layoutResizeObserver = new ResizeObserver(() => clampLayoutIfLocked())
    layoutResizeObserver.observe(el)
  })
  try {
    await ensureAuth()
  } catch (_) {
    return
  }
  await loadHistory()
})

onBeforeUnmount(() => {
  if (onLayoutWindowResize) window.removeEventListener('resize', onLayoutWindowResize)
  clearTimeout(layoutResizeDebounceTimer)
  teardownRouteCardResizeObserver()
  if (layoutResizeObserver) {
    layoutResizeObserver.disconnect()
    layoutResizeObserver = null
  }
})
</script>

<template>
  <div class="container">
    <div class="card top-card">
      <div class="title-hero">
        <div ref="highwaySceneRef" class="highway-scene">
          <div class="sky-layer">
            <div class="cloud c1"></div>
            <div class="cloud c2"></div>
            <div class="cloud c3"></div>
            <div class="tree t1"><div class="leaf"></div><div class="trunk"></div></div>
            <div class="tree t2"><div class="leaf"></div><div class="trunk"></div></div>
            <div class="tree t3"><div class="leaf"></div><div class="trunk"></div></div>
            <div class="tree t4"><div class="leaf"></div><div class="trunk"></div></div>
            <div class="celestial sun-pos"><div class="sun"></div></div>
            <div class="celestial moon-pos"><div class="moon"></div></div>
          </div>
          <div class="title-overlay">
            <h1 class="title">智慧交通客服智能体</h1>
          </div>
          <div class="road"></div>
          <div ref="trafficLayerRef" class="traffic-layer"></div>
          <div ref="carRef" class="car">
            <div class="car-body"></div>
            <div class="car-top"></div>
            <div class="headlight top"></div>
            <div class="headlight bottom"></div>
            <div class="wheel left"></div>
            <div class="wheel right"></div>
          </div>
        </div>
      </div>
      <div class="top-actions">
        <button class="ghost-btn theme-toggle" :class="isDayTheme ? 'day' : 'night'" @click="toggleTheme">
          {{ isDayTheme ? '切换到夜间模式' : '切换到白天模式' }}
        </button>
      </div>
    </div>

    <div ref="layoutRowRef" class="layout">
      <div ref="routeCardRef" class="card route-card">
        <div class="route-header">
          <h2 class="route-title">路径规划</h2>
          <span class="badge">{{ routeBadge }}</span>
        </div>
        <div class="route-meta">
          <div class="meta-item">
            <div class="k">预计里程</div>
            <div class="v">{{ routeDistanceText }}</div>
          </div>
          <div class="meta-item">
            <div class="k">预计时长</div>
            <div class="v">{{ routeDurationText }}</div>
          </div>
        </div>
        <div class="route-body">
          <div class="waypoints">
            <div class="place">起点：{{ routeOriginText }}</div>
            <div class="arrow">→</div>
            <div class="place">终点：{{ routeDestinationText }}</div>
          </div>
          <div class="map-shell">
            <div ref="mapRef" class="route-map"></div>
            <div class="map-tip">左侧地图展示规划路径折线（滚轮不缩放地图，避免影响列表与对话区滚动；请用左上角 +/- 缩放）</div>
          </div>
          <div class="routes-list">
            <template v-for="(seg, si) in routeSegments" :key="seg.key">
              <div v-if="routeResult?.multi_segment && seg.title !== '全程'" class="route-leg-label">{{ seg.title }}</div>
              <div class="route-section">
                <h4>途经高速</h4>
                <div class="highway-rich-grid">
                  <template v-if="!seg.highways.length">
                    <div class="highway-rich-card muted">暂无高速信息</div>
                  </template>
                  <div
                    v-for="(hw, hi) in highwayCardItems(seg.highways)"
                    :key="`${seg.key}-hw-${hi}-${hw.raw}`"
                    class="highway-rich-card"
                  >
                    <div class="hw-code">{{ hw.code }}</div>
                    <div class="hw-sub">{{ hw.subtitle }}</div>
                  </div>
                </div>
              </div>
            </template>
            <div v-if="tripHints.length" class="route-section trip-hints-section">
              <h4>行程提示</h4>
              <div class="trip-hint-grid">
                <div v-for="(th, ti) in tripHints" :key="ti" class="trip-hint-card">
                  <div class="trip-hint-title">{{ th.title }}</div>
                  <div class="trip-hint-detail">{{ th.detail }}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="card chat-card">
        <div class="chat-header">
          <h2 class="chat-title">智能对话</h2>
          <div class="chat-header-actions">
            <span class="badge">用户：{{ currentUsername || '未登录' }}</span>
            <button class="tiny-btn" @click="newChat">新建对话</button>
            <button class="tiny-btn" @click="clearChat">清空当前</button>
            <button class="tiny-btn" @click="logout">退出登录</button>
          </div>
        </div>
        <div ref="chatBoxRef" class="chat-box">
          <div
            v-for="(msg, idx) in messages"
            :key="`${idx}-${msg.role}-${msg.text?.slice(0, 24)}`"
            class="msg-block"
            :class="{ 'msg-block--user': msg.role === 'user' }"
          >
            <div class="msg" :class="msg.role">{{ msg.text }}</div>
            <div v-if="msg.role === 'agent' && msg.followUps?.length" class="follow-ups">
              <div class="follow-ups-label">你可能还想问（点击将发送该句，由助手按当前能力作答）</div>
              <div class="follow-ups-chips">
                <button
                  v-for="(fu, fi) in msg.followUps"
                  :key="fi"
                  type="button"
                  class="follow-up-chip"
                  :disabled="sending"
                  @click="submitFollowUpQuestion(idx, fi)"
                >
                  {{ fu.question }}
                </button>
              </div>
            </div>
          </div>
        </div>
        <div class="composer">
          <textarea v-model="inputMessage" placeholder="例如：北京到秦皇岛；有交通事故吗？" @keydown="onInputKeydown"></textarea>
          <button class="send-btn" :disabled="sending" @click="sendMessage">发送</button>
        </div>
      </div>
    </div>
  </div>
</template>
