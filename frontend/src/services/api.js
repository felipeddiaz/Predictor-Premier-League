const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

async function request(method, path, body = null) {
  const options = {
    method,
    headers: { 'Content-Type': 'application/json' },
  }
  if (body) {
    options.body = JSON.stringify(body)
  }

  const response = await fetch(`${API_BASE_URL}${path}`, options)

  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(error.detail || `Request failed: ${response.status}`)
  }

  return response.json()
}

export const api = {
  // Health check
  health: () => request('GET', '/health'),

  // List available teams
  teams: () => request('GET', '/teams'),

  // Simple prediction (only team names)
  predict: (homeTeam, awayTeam) =>
    request('POST', '/predict', {
      home_team: homeTeam,
      away_team: awayTeam,
    }),

  // Detailed prediction with odds (value betting analysis)
  predictWithOdds: (homeTeam, awayTeam, odds) =>
    request('POST', '/predictions/detail', {
      local: homeTeam,
      visitante: awayTeam,
      cuota_h: odds.cuota_h,
      cuota_d: odds.cuota_d,
      cuota_a: odds.cuota_a,
    }),
}

export default api
