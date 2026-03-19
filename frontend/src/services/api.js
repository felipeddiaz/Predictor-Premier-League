const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

async function request(method, path, body = null) {
  const opts = {
    method,
    headers: { 'Content-Type': 'application/json' },
    signal: AbortSignal.timeout(15000),
  }
  if (body) opts.body = JSON.stringify(body)

  const res = await fetch(`${API_BASE}${path}`, opts)
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Request failed: ${res.status}`)
  }
  return res.json()
}

export const api = {
  health: () => request('GET', '/health'),
  teams: () => request('GET', '/teams'),
  status: () => request('GET', '/status'),

  predictDetail: (home, away, oddsH, oddsD, oddsA) =>
    request('POST', '/predictions/detail', {
      local: home,
      visitante: away,
      cuota_h: oddsH,
      cuota_d: oddsD,
      cuota_a: oddsA,
    }),

  predictSimple: (home, away) =>
    request('POST', '/predict', {
      home_team: home,
      away_team: away,
    }),

  teamStats: (team, isLocal = true) =>
    request('GET', `/teams/${encodeURIComponent(team)}/stats?is_local=${isLocal}`),

  validateOdds: (h, d, a) =>
    request('POST', `/utils/validate-odds?cuota_h=${h}&cuota_d=${d}&cuota_a=${a}`),

  expectedValue: (prob, odds) =>
    request('GET', `/utils/expected-value?prob_model=${prob}&cuota=${odds}`),
}
