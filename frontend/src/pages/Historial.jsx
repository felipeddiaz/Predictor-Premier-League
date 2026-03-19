import { useState, useMemo, useCallback } from 'react'
import {
  getHistory, clearHistory, updateHistoryResult,
  calcEV, calcKelly, calcEdge, impliedProbs,
} from '../utils/calc'

const PER_PAGE = 20
const PRED_LABELS = { H: 'Local', D: 'Empate', A: 'Visitante' }
const RESULT_LABELS = { H: '1', D: 'X', A: '2' }

function exportCSV(data) {
  const cols = ['fecha', 'local', 'visitante', 'pred', 'P(H)', 'P(D)', 'P(A)', 'EV', 'VB', 'kelly', 'resultado_real', 'correcto']
  const rows = data.map(h => [
    new Date(h.timestamp).toLocaleDateString(),
    h.home, h.away,
    PRED_LABELS[h.prediction] || h.prediction,
    h.probH, h.probD, h.probA,
    h.ev, h.isVB ? 'Si' : 'No',
    h.kelly > 0 ? `${h.kelly}%` : '',
    h.actual_result ? RESULT_LABELS[h.actual_result] || h.actual_result : '',
    h.actual_result ? (h.actual_result === h.prediction ? 'Si' : 'No') : '',
  ])
  const csv = [cols, ...rows].map(r => r.join(',')).join('\n')
  const a = document.createElement('a')
  a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }))
  a.download = 'predicciones_pl.csv'
  a.click()
}

/* ── Bankroll simulation using Kelly staking on VB bets with recorded results ── */
function buildBankroll(history) {
  const bk = 100
  const vbWithResults = history
    .filter(h => h.isVB && h.actual_result && h.kelly > 0 && h.oddsH)
    .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp))

  if (!vbWithResults.length) return null

  const points = [{ x: 0, y: bk, label: 'Inicio' }]
  let current = bk
  let won = 0, lost = 0

  for (const h of vbWithResults) {
    const stakeAmt = current * (h.kelly / 100)
    const correct = h.actual_result === h.prediction
    const oddsMap = { H: h.oddsH || 2.5, D: h.oddsD || 3.4, A: h.oddsA || 2.5 }
    const predOdds = oddsMap[h.prediction] || 2.5
    if (correct) {
      current += stakeAmt * (predOdds - 1)
      won++
    } else {
      current -= stakeAmt
      lost++
    }
    points.push({ x: points.length, y: +current.toFixed(2), label: `${h.home} v ${h.away}` })
  }

  return { points, current: +current.toFixed(2), won, lost, total: won + lost }
}

/* ── Mini SVG line chart ── */
function BankrollChart({ points }) {
  if (!points || points.length < 2) return null
  const W = 300, H = 60, pad = 4
  const ys = points.map(p => p.y)
  const minY = Math.min(...ys)
  const maxY = Math.max(...ys)
  const rangeY = maxY - minY || 1
  const rangeX = points.length - 1

  const px = i => pad + (i / rangeX) * (W - pad * 2)
  const py = y => H - pad - ((y - minY) / rangeY) * (H - pad * 2)

  const path = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${px(i).toFixed(1)},${py(p.y).toFixed(1)}`).join(' ')
  const area = `${path} L${px(points.length - 1).toFixed(1)},${H} L${px(0).toFixed(1)},${H} Z`

  const isProfit = points[points.length - 1].y >= points[0].y
  const color = isProfit ? 'var(--grn)' : 'var(--red)'
  const fillColor = isProfit ? 'rgba(22,163,74,.12)' : 'rgba(220,38,38,.10)'

  return (
    <div className="bk-chart">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
        <path d={area} fill={fillColor} />
        <path d={path} fill="none" stroke={color} strokeWidth="1.5" />
        {[0, points.length - 1].map(i => (
          <circle key={i} cx={px(i)} cy={py(points[i].y)} r="2.5" fill={color} />
        ))}
      </svg>
    </div>
  )
}

export default function Historial() {
  const [history, setHistory] = useState(() => getHistory())
  const [filter, setFilter] = useState('all')
  const [sort, setSort] = useState({ col: 'timestamp', asc: false })
  const [page, setPage] = useState(1)

  const reload = useCallback(() => setHistory(getHistory()), [])

  const enriched = useMemo(() => history.map(h => {
    const bh = h.oddsH || 2.10, bd = h.oddsD || 3.40, ba = h.oddsA || 3.60
    const ip = impliedProbs(bh, bd, ba)
    const probMap = { H: h.probH, D: h.probD, A: h.probA }
    const oddsMap = { H: bh, D: bd, A: ba }
    const mktMap = { H: ip.home, D: ip.draw, A: ip.away }
    const predProb = probMap[h.prediction] || 0
    const predOdds = oddsMap[h.prediction] || 1
    const predMkt = mktMap[h.prediction] || 0
    const ev = calcEV(predProb, predOdds)
    const edge = calcEdge(predProb, predMkt)
    const kelly = calcKelly(predProb, predOdds)
    const isVB = edge >= 10 && predOdds <= 5 && predProb >= 35
    return { ...h, ev, edge, kelly, isVB }
  }), [history])

  const filtered = useMemo(() => {
    const filters = {
      all: () => true, vb: r => r.isVB,
      H: r => r.prediction === 'H', D: r => r.prediction === 'D', A: r => r.prediction === 'A',
      correct: r => r.actual_result && r.actual_result === r.prediction,
      wrong: r => r.actual_result && r.actual_result !== r.prediction,
    }
    const data = enriched.filter(filters[filter] || filters.all)
    data.sort((a, b) => {
      const av = a[sort.col] ?? '', bv = b[sort.col] ?? ''
      return sort.asc ? (av > bv ? 1 : -1) : (av < bv ? 1 : -1)
    })
    return data
  }, [enriched, filter, sort])

  const totalPages = Math.ceil(filtered.length / PER_PAGE)
  const pageData = filtered.slice((page - 1) * PER_PAGE, page * PER_PAGE)

  // KPIs
  const total = enriched.length
  const vbCount = enriched.filter(h => h.isVB).length
  const avgConf = total > 0 ? (enriched.reduce((s, h) => s + (h.confidence || 0), 0) / total * 100).toFixed(0) : 0
  const withResults = enriched.filter(h => h.actual_result)
  const correct = withResults.filter(h => h.actual_result === h.prediction).length
  const accuracy = withResults.length > 0 ? (correct / withResults.length * 100).toFixed(0) : null

  // P&L — simulated: +kelly*(odds-1) on win, -kelly on loss, only VB bets
  const pnl = enriched.reduce((sum, h) => {
    if (!h.isVB || !h.actual_result || !h.kelly) return sum
    const oddsMap = { H: h.oddsH || 2.5, D: h.oddsD || 3.4, A: h.oddsA || 2.5 }
    const predOdds = oddsMap[h.prediction] || 2.5
    return sum + (h.actual_result === h.prediction
      ? h.kelly * (predOdds - 1)
      : -h.kelly)
  }, 0)

  const bkSim = useMemo(() => buildBankroll(enriched), [enriched])

  const handleResult = useCallback((id, result) => {
    const entry = history.find(h => h.id === id)
    const newVal = entry?.actual_result === result ? null : result // toggle off
    updateHistoryResult(id, newVal)
    reload()
  }, [history, reload])

  const handleSort = col => {
    setSort(prev => prev.col === col ? { col, asc: !prev.asc } : { col, asc: false })
    setPage(1)
  }
  const handleFilter = f => { setFilter(f); setPage(1) }
  const handleClear = () => {
    if (confirm('\u00BFBorrar todo el historial?')) { clearHistory(); setHistory([]) }
  }

  return (
    <div className="pg-hist">
      <div className="hist-wrap">
        {/* KPI SIDEBAR */}
        <div className="hist-side">
          <div className="hs-kpi">
            <div className="hs-kpi-val" style={{ color: 'var(--ink)' }}>{total}</div>
            <div className="hs-kpi-lbl">Predicciones</div>
          </div>
          <div className="hs-kpi">
            <div className="hs-kpi-val" style={{ color: 'var(--amb)' }}>{vbCount}</div>
            <div className="hs-kpi-lbl">Value bets</div>
          </div>
          <div className="hs-kpi">
            <div className="hs-kpi-val" style={{ color: 'var(--blue)' }}>{avgConf}%</div>
            <div className="hs-kpi-lbl">Confianza prom.</div>
          </div>
          {accuracy !== null && (
            <div className="hs-kpi">
              <div className="hs-kpi-val" style={{ color: +accuracy >= 50 ? 'var(--grn)' : 'var(--red)' }}>
                {accuracy}%
              </div>
              <div className="hs-kpi-lbl">Acierto ({correct}/{withResults.length})</div>
            </div>
          )}
          {withResults.length > 0 && (
            <div className="hs-kpi">
              <div className="hs-kpi-val" style={{ color: pnl >= 0 ? 'var(--grn)' : 'var(--red)', fontSize: 16 }}>
                {pnl >= 0 ? '+' : ''}{pnl.toFixed(1)}u
              </div>
              <div className="hs-kpi-lbl">P&L (VB bets)</div>
            </div>
          )}

          {/* Bankroll simulation */}
          {bkSim ? (
            <div className="bk-section">
              <div className="bk-title">Simulaci\u00F3n bankroll</div>
              <div className="bk-kpis">
                <div className="bk-kpi">
                  <div className="bk-kpi-val" style={{ color: bkSim.current >= 100 ? 'var(--grn)' : 'var(--red)' }}>
                    {bkSim.current}u
                  </div>
                  <div className="bk-kpi-lbl">Final (de 100u)</div>
                </div>
                <div className="bk-kpi">
                  <div className="bk-kpi-val" style={{ color: bkSim.current >= 100 ? 'var(--grn)' : 'var(--red)' }}>
                    {bkSim.current >= 100 ? '+' : ''}{(bkSim.current - 100).toFixed(1)}%
                  </div>
                  <div className="bk-kpi-lbl">Retorno</div>
                </div>
              </div>
              <div style={{ fontSize: 9, color: 'var(--ink4)', marginBottom: 5, textAlign: 'center' }}>
                {bkSim.won}V / {bkSim.lost}D — {bkSim.total} apuestas VB
              </div>
              <BankrollChart points={bkSim.points} />
            </div>
          ) : withResults.length === 0 && vbCount > 0 ? (
            <div className="bk-section">
              <div className="bk-title">Simulaci\u00F3n bankroll</div>
              <div className="bk-empty">Registra resultados reales en la tabla para simular el bankroll</div>
            </div>
          ) : null}

          {total > 0 && (
            <button onClick={handleClear} style={{
              background: 'var(--red-l)', border: '1px solid var(--red-b)', borderRadius: 7,
              color: 'var(--red)', fontSize: 11, fontWeight: 600, padding: '6px 10px',
              cursor: 'pointer', fontFamily: 'var(--sans)',
            }}>
              Borrar historial
            </button>
          )}
        </div>

        {/* MAIN TABLE */}
        <div className="hist-main">
          <div className="hist-bar">
            {[
              ['all', 'Todos'],
              ['vb', 'Value bets'],
              ['H', 'Local'],
              ['D', 'Empate'],
              ['A', 'Visitante'],
              ['correct', '\u2705 Correcto'],
              ['wrong', '\u274C Fallido'],
            ].map(([key, label]) => (
              <button key={key} className={`flt${filter === key ? ' on' : ''}`}
                onClick={() => handleFilter(key)}>
                {label}
              </button>
            ))}
            {total > 0 && (
              <button className="btn-export" onClick={() => exportCSV(enriched)}>
                Exportar CSV
              </button>
            )}
          </div>

          <div className="hist-table-wrap">
            {!history.length ? (
              <div style={{ padding: 40, textAlign: 'center', color: 'var(--ink4)' }}>
                <div style={{ fontSize: 32, opacity: .3, marginBottom: 8 }}>{'\uD83D\uDCCA'}</div>
                <div style={{ fontSize: 13, fontWeight: 500 }}>
                  A{'\u00FA'}n no hay predicciones. Usa el Predictor para generar tu historial.
                </div>
              </div>
            ) : (
              <table className="ht">
                <thead>
                  <tr>
                    <th onClick={() => handleSort('jornada')}>J</th>
                    <th onClick={() => handleSort('home')}>Local</th>
                    <th onClick={() => handleSort('away')}>Visitante</th>
                    <th onClick={() => handleSort('prediction')}>Pred.</th>
                    <th onClick={() => handleSort('probH')}>P(H)</th>
                    <th onClick={() => handleSort('probD')}>P(D)</th>
                    <th onClick={() => handleSort('probA')}>P(A)</th>
                    <th onClick={() => handleSort('ev')}>EV</th>
                    <th onClick={() => handleSort('isVB')}>VB</th>
                    <th onClick={() => handleSort('kelly')}>Kelly</th>
                    <th title="Registra el resultado real del partido">Resultado real</th>
                    <th onClick={() => handleSort('timestamp')}>Fecha</th>
                  </tr>
                </thead>
                <tbody>
                  {pageData.map(h => {
                    const correct = h.actual_result && h.actual_result === h.prediction
                    const wrong = h.actual_result && h.actual_result !== h.prediction
                    return (
                      <tr key={h.id} style={correct ? { background: 'var(--grn-l)' } : wrong ? { background: 'var(--red-l)' } : {}}>
                        <td style={{ color: 'var(--ink4)', fontWeight: 600, fontFamily: 'var(--mono)' }}>
                          J{h.jornada}
                        </td>
                        <td style={{ fontWeight: 600 }}>{h.home}</td>
                        <td style={{ color: 'var(--ink3)' }}>{h.away}</td>
                        <td>
                          <span className={`bgt bgt-${h.prediction}`}>
                            {PRED_LABELS[h.prediction]}
                          </span>
                          {correct && <span className="res-correct" style={{ marginLeft: 4 }}>{'\u2713'}</span>}
                          {wrong && <span className="res-wrong" style={{ marginLeft: 4 }}>{'\u2715'}</span>}
                        </td>
                        <td style={{ fontFamily: 'var(--mono)' }}>{h.probH}%</td>
                        <td style={{ fontFamily: 'var(--mono)' }}>{h.probD}%</td>
                        <td style={{ fontFamily: 'var(--mono)' }}>{h.probA}%</td>
                        <td style={{ fontFamily: 'var(--mono)', color: h.ev >= 0 ? 'var(--grn)' : 'var(--red)' }}>
                          {h.ev >= 0 ? '+' : ''}{h.ev}
                        </td>
                        <td>
                          {h.isVB
                            ? <span className="bgt bgt-vb">{'\u2605'} VB</span>
                            : <span style={{ color: 'var(--ink5)' }}>{'\u2014'}</span>}
                        </td>
                        <td style={{ fontFamily: 'var(--mono)', color: h.kelly > 0 ? 'var(--amb)' : 'var(--ink5)' }}>
                          {h.kelly > 0 ? `${h.kelly}%` : '\u2014'}
                        </td>
                        <td>
                          <div className="res-btns">
                            {['H', 'D', 'A'].map(r => (
                              <button
                                key={r}
                                className={`res-btn${h.actual_result === r ? ` sel-${r}` : ''}`}
                                onClick={() => handleResult(h.id, r)}
                                title={PRED_LABELS[r]}
                              >
                                {RESULT_LABELS[r]}
                              </button>
                            ))}
                          </div>
                        </td>
                        <td style={{ fontSize: 10, color: 'var(--ink4)' }}>
                          {new Date(h.timestamp).toLocaleDateString()}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            )}
          </div>

          {totalPages > 1 && (
            <div className="pg-row">
              {Array.from({ length: totalPages }, (_, i) => (
                <button key={i} className={`pgb${i + 1 === page ? ' on' : ''}`}
                  onClick={() => setPage(i + 1)}>
                  {i + 1}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
