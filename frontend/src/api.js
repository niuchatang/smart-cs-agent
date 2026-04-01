export async function apiGet(url) {
  const resp = await fetch(url, { credentials: 'same-origin' })
  const data = await resp.json().catch(() => ({}))
  if (!resp.ok) {
    const detail = data?.detail || JSON.stringify(data) || '请求失败'
    throw new Error(detail)
  }
  return data
}

export async function apiPost(url, payload) {
  const resp = await fetch(url, {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload ?? {}),
  })
  const data = await resp.json().catch(() => ({}))
  if (!resp.ok) {
    const detail = data?.detail || JSON.stringify(data) || '请求失败'
    throw new Error(detail)
  }
  return data
}

export async function apiDelete(url) {
  const resp = await fetch(url, { method: 'DELETE', credentials: 'same-origin' })
  const data = await resp.json().catch(() => ({}))
  if (!resp.ok) {
    const detail = data?.detail || JSON.stringify(data) || '请求失败'
    throw new Error(detail)
  }
  return data
}
