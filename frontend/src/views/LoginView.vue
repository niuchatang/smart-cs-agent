<script setup>
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { apiGet, apiPost } from '../api'

const router = useRouter()
const mode = ref('login')
const username = ref('')
const password = ref('')
const captchaInput = ref('')
const captchaText = ref('')
const captchaCanvas = ref(null)
const submitting = ref(false)
const msgText = ref('')
const msgType = ref('')

function setMsg(text, type = '') {
  msgText.value = text
  msgType.value = type
}

function makeCaptchaText() {
  const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz23456789'
  let out = ''
  for (let i = 0; i < 4; i += 1) out += chars[Math.floor(Math.random() * chars.length)]
  return out
}

function redrawCaptcha() {
  captchaText.value = makeCaptchaText()
  const canvas = captchaCanvas.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  ctx.fillStyle = '#f8fafc'
  ctx.fillRect(0, 0, canvas.width, canvas.height)
  for (let i = 0; i < 5; i += 1) {
    ctx.strokeStyle = `rgba(59,130,246,${0.2 + Math.random() * 0.25})`
    ctx.beginPath()
    ctx.moveTo(Math.random() * 120, Math.random() * 42)
    ctx.lineTo(Math.random() * 120, Math.random() * 42)
    ctx.stroke()
  }
  for (let i = 0; i < 20; i += 1) {
    ctx.fillStyle = `rgba(30,64,175,${0.15 + Math.random() * 0.35})`
    ctx.fillRect(Math.random() * 120, Math.random() * 42, 1.6, 1.6)
  }
  ctx.font = "bold 24px 'Courier New', monospace"
  ctx.textBaseline = 'middle'
  for (let i = 0; i < captchaText.value.length; i += 1) {
    const ch = captchaText.value[i]
    const x = 18 + i * 22
    const y = 21 + (Math.random() * 7 - 3.5)
    const angle = (Math.random() * 24 - 12) * (Math.PI / 180)
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(angle)
    ctx.fillStyle = `rgb(${30 + Math.floor(Math.random() * 80)}, ${64 + Math.floor(Math.random() * 60)}, ${120 + Math.floor(Math.random() * 80)})`
    ctx.fillText(ch, -6, 0)
    ctx.restore()
  }
}

function switchMode(nextMode) {
  mode.value = nextMode
  captchaInput.value = ''
  setMsg('')
  redrawCaptcha()
}

async function submitAuth() {
  if (!username.value || !password.value || !captchaInput.value) {
    setMsg('请输入用户名、密码和验证码', 'err')
    return
  }
  if (captchaInput.value.toLowerCase() !== captchaText.value.toLowerCase()) {
    setMsg('验证码错误，请重试', 'err')
    captchaInput.value = ''
    redrawCaptcha()
    return
  }
  if (mode.value === 'register' && !/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$/.test(password.value)) {
    setMsg('注册密码需>=8位，且包含大小写字母和数字', 'err')
    return
  }
  submitting.value = true
  setMsg(mode.value === 'login' ? '登录中...' : '注册中...')
  try {
    const endpoint = mode.value === 'login' ? '/auth/login' : '/auth/register'
    await apiPost(endpoint, {
      username: username.value.trim(),
      password: password.value,
    })
    setMsg('成功，正在进入系统...', 'ok')
    router.replace('/')
  } catch (err) {
    setMsg(String(err), 'err')
    redrawCaptcha()
  } finally {
    submitting.value = false
  }
}

onMounted(async () => {
  redrawCaptcha()
  try {
    await apiGet('/auth/me')
    router.replace('/')
  } catch (_) {}
})
</script>

<template>
  <div class="login-page">
    <div class="auth-card">
      <h1>智慧交通客服智能体</h1>
      <p class="sub">先登录后使用路线规划与对话功能</p>
      <div class="tabs">
        <button class="tab" :class="{ active: mode === 'login' }" type="button" @click="switchMode('login')">登录</button>
        <button class="tab" :class="{ active: mode === 'register' }" type="button" @click="switchMode('register')">注册</button>
      </div>
      <form @submit.prevent="submitAuth">
        <div class="field">
          <label>用户名</label>
          <input v-model.trim="username" autocomplete="username" placeholder="3-32位：字母/数字/_/-" />
        </div>
        <div class="field">
          <label>密码</label>
          <input
            v-model="password"
            type="password"
            :autocomplete="mode === 'login' ? 'current-password' : 'new-password'"
            :placeholder="mode === 'login' ? '请输入密码' : '至少8位，包含大小写字母和数字'"
          />
          <div class="hint">{{ mode === 'register' ? '注册密码要求：>=8位，且包含大写字母、小写字母、数字。' : '' }}</div>
        </div>
        <div class="field">
          <label>图形验证码</label>
          <div class="captcha-row">
            <input v-model.trim="captchaInput" autocomplete="off" placeholder="输入验证码" />
            <canvas ref="captchaCanvas" class="captcha-canvas" width="120" height="42" @click="redrawCaptcha"></canvas>
            <button class="captcha-refresh" type="button" @click="redrawCaptcha">刷新</button>
          </div>
        </div>
        <button class="submit-btn" :disabled="submitting" type="submit">{{ mode === 'login' ? '登录' : '注册并登录' }}</button>
      </form>
      <div class="msg" :class="msgType">{{ msgText }}</div>
    </div>
  </div>
</template>
