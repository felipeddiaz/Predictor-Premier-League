export function teamInitials(name) {
  return name.split(' ').map(w => w[0]).join('').slice(0, 3)
}

export function impliedProbs(h, d, a) {
  const rawH = 1 / h, rawD = 1 / d, rawA = 1 / a
  const total = rawH + rawD + rawA
  return {
    home: +(rawH / total * 100).toFixed(1),
    draw: +(rawD / total * 100).toFixed(1),
    away: +(rawA / total * 100).toFixed(1),
    overround: +((total - 1) * 100).toFixed(1),
    vig: +(total * 100).toFixed(1),
  }
}

export function calcEdge(modelProb, marketProb) {
  return +(modelProb - marketProb).toFixed(1)
}

export function calcEV(prob, odds) {
  return +(prob / 100 * (odds - 1) - (1 - prob / 100)).toFixed(3)
}

export function calcKelly(prob, odds, fraction = 0.25, max = 0.025) {
  const b = odds - 1
  const k = (prob / 100 * b - (1 - prob / 100)) / b
  return +(Math.min(Math.max(0, k * fraction), max) * 100).toFixed(2)
}

export function isValueBet(edge, odds, prob) {
  return edge >= 10 && odds <= 5 && prob >= 35
}

export function entropyConfidence(ph, pd, pa) {
  const ps = [ph / 100, pd / 100, pa / 100]
  const ent = -ps.reduce((s, p) => s + (p > 0 ? p * Math.log2(p) : 0), 0)
  const conf = Math.round((1 - ent / Math.log2(3)) * 100)
  return {
    value: conf,
    label: conf >= 65 ? 'Alta' : conf >= 40 ? 'Media' : 'Baja',
    color: conf >= 65 ? 'var(--grn)' : conf >= 40 ? 'var(--amb)' : 'var(--red)',
  }
}

export function parseForm(formStr) {
  if (!formStr) return []
  const dots = []
  formStr.split('-').forEach(part => {
    const count = parseInt(part)
    const letter = part.replace(/\d/g, '')
    if (!isNaN(count) && letter) {
      for (let i = 0; i < count; i++) dots.push(letter)
    }
  })
  return dots
}

export function recommendation(edge, ev, kelly, odds) {
  const e = parseFloat(edge)
  if (e >= 10 && odds <= 5 && ev > 0) return { icon: '\u2705', label: 'Value bet', cls: 'vb', stake: kelly }
  if (e < -5) return { icon: '\u274C', label: 'Sobrevalorado', cls: 'over', stake: 0 }
  return { icon: '\u26A0\uFE0F', label: 'No bet', cls: '', stake: 0 }
}

export function mapPrediction(data, oddsH, oddsD, oddsA) {
  const ph = +(data.prob_local * 100).toFixed(1)
  const pd = +(data.prob_draw * 100).toFixed(1)
  const pa = +(data.prob_away * 100).toFixed(1)

  const mktH = data.market_prob_local != null ? +(data.market_prob_local * 100).toFixed(1) : null
  const mktD = data.market_prob_draw != null ? +(data.market_prob_draw * 100).toFixed(1) : null
  const mktA = data.market_prob_away != null ? +(data.market_prob_away * 100).toFixed(1) : null

  const predMap = { 'Local': 'H', 'Empate': 'D', 'Visitante': 'A', 'Home Win': 'H', 'Draw': 'D', 'Away Win': 'A' }

  return {
    probH: ph, probD: pd, probA: pa,
    mktH, mktD, mktA,
    prediction: predMap[data.prediction || data.predicted_outcome] || 'H',
    confidence: data.confidence,
    formHome: parseForm(data.form_local || data.home_form),
    formAway: parseForm(data.form_away || data.away_form),
    over25: data.over25_prob != null ? +(data.over25_prob * 100).toFixed(1) : null,
    over35cards: data.over35_cards_prob != null ? +(data.over35_cards_prob * 100).toFixed(1) : null,
    over95corners: data.over95_corners_prob != null ? +(data.over95_corners_prob * 100).toFixed(1) : null,
    source: 'api',
  }
}

// localStorage helpers for historial
const HISTORY_KEY = 'pl_predictor_history'

export function saveToHistory(entry) {
  const hist = getHistory()
  hist.unshift({ ...entry, id: Date.now(), timestamp: new Date().toISOString() })
  if (hist.length > 500) hist.length = 500
  localStorage.setItem(HISTORY_KEY, JSON.stringify(hist))
}

export function getHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]')
  } catch { return [] }
}

export function clearHistory() {
  localStorage.removeItem(HISTORY_KEY)
}
