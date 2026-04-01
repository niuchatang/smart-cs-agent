/** 常见编号 → 习惯称呼（与后端 main._highway_name_from_code 对齐并扩展） */
export const HIGHWAY_NAME_BY_CODE = {
  G1: '京哈高速',
  G2: '京沪高速',
  G3: '京台高速',
  G4: '京港澳高速',
  G5: '京昆高速',
  G6: '京藏高速',
  G7: '京新高速',
  G11: '鹤大高速',
  G15: '沈海高速',
  G18: '荣乌高速',
  G20: '青银高速',
  G22: '青兰高速',
  G25: '长深高速',
  G30: '连霍高速',
  G35: '济广高速',
  G36: '宁洛高速',
  G40: '沪陕高速',
  G42: '沪蓉高速',
  G45: '大广高速',
  G50: '沪渝高速',
  G56: '杭瑞高速',
  G60: '沪昆高速',
  G65: '包茂高速',
  G69: '银百高速',
  G70: '福银高速',
  G75: '兰海高速',
  G76: '厦蓉高速',
  G78: '汕昆高速',
  G80: '广昆高速',
  G0111: '秦滨高速',
  G0211: '津汕高速',
  G0421: '许广高速',
  G0424: '武汉都市圈环线高速',
  G1511: '日兰高速',
  G2513: '淮徐高速',
  G3511: '菏宝高速',
  G4213: '麻安高速',
  G4221: '沪武高速',
  G4224: '武天宜高速',
  G4512: '奈营高速',
  G5012: '恩广高速',
  G5013: '渝蓉高速',
  G5021: '石渝高速',
  G5512: '晋新高速',
  G5513: '长张高速',
  G7011: '十天高速',
  S1: '机场高速',
  S2: '沪芦高速',
  S7: '沪崇高速',
  S15: '汉蔡高速',
  S25: '静兴高速',
  S81: '鄂东环线',
  S86: '赤洪高速',
  S8105: '济乐高速',
}

/**
 * 将途经高速字符串拆成「编号 + 说明」，便于卡片展示。
 */
export function enrichHighway(raw) {
  const s = String(raw || '').trim()
  if (!s) return { code: '', subtitle: '', raw: s }
  const m = s.match(/^([GS]\d{1,4})(.*)$/i)
  if (m) {
    const code = m[1].toUpperCase()
    let rest = (m[2] || '').replace(/^[,，\s]+/, '')
    if (rest && /[\u4e00-\u9fff]/.test(rest)) {
      return { code, subtitle: rest, raw: s }
    }
    const known = HIGHWAY_NAME_BY_CODE[code]
    return { code, subtitle: known || '国家/省域高速公路网', raw: s }
  }
  return { code: s, subtitle: s.includes('高速') ? '收费高速公路' : '道路路段', raw: s }
}

/**
 * 多段路线用 leg_details；否则合并为一段「全程」。
 */
export function buildRouteSegments(routeResult) {
  if (!routeResult) return []
  const legs = routeResult.leg_details
  if (Array.isArray(legs) && legs.length) {
    return legs.map((ld, i) => ({
      key: `leg-${i}`,
      title: `${ld.from || ''} → ${ld.to || ''}`.trim() || `第 ${i + 1} 段`,
      highways: Array.isArray(ld.highways) ? ld.highways : [],
    }))
  }
  return [
    {
      key: 'all',
      title: '全程',
      highways: Array.isArray(routeResult.highways) ? routeResult.highways : [],
    },
  ]
}
