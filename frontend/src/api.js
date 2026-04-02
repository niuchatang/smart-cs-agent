const fetchOpts = { credentials: 'include' }

/** 将 FastAPI 422 的 detail 数组转成可读文案 */
function formatApiError(data) {
  const d = data?.detail
  if (Array.isArray(d)) {
    return d
      .map((e) => {
        const loc = Array.isArray(e.loc) ? e.loc.filter((x) => x !== 'body').join('.') : ''
        const msg = e.msg || ''
        return loc ? `${loc}: ${msg}` : msg
      })
      .filter(Boolean)
      .join('；') || '请求参数校验失败'
  }
  if (typeof d === 'string') return d
  return JSON.stringify(data || {})
}

export async function apiGet(url) {
  const resp = await fetch(url, fetchOpts)
  const data = await resp.json().catch(() => ({}))
  if (!resp.ok) {
    throw new Error(formatApiError(data))
  }
  return data
}

export async function apiPost(url, payload) {
  const resp = await fetch(url, {
    method: 'POST',
    ...fetchOpts,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload ?? {}),
  })
  const data = await resp.json().catch(() => ({}))
  if (!resp.ok) {
    throw new Error(formatApiError(data))
  }
  return data
}

export async function apiDelete(url) {
  const resp = await fetch(url, { method: 'DELETE', ...fetchOpts })
  const data = await resp.json().catch(() => ({}))
  if (!resp.ok) {
    throw new Error(formatApiError(data))
  }
  return data
}
