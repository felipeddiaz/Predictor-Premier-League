import { useState, useMemo } from 'react'
import { getHistory, clearHistory, calcEV, calcKelly, calcEdge, impliedProbs } from '../utils/calc'

const PER_PAGE = 20
const PRED_LABELS = { H: 'Local', D: 'Empate', A: 'Visitante' }

export default function Historial() {
  const [history, setHistory] = useState(() => getHistory())
  const [filter, setFilter] = useState('all')
  const [sort, setSort] = useState({ col: 'timestamp', asc: false })
  const [page, setPage] = useState(1)

  const filtered = useMemo(() => {
    let data = [...history]

    // Enrich with calculated fields
    data = data.map(h => {
      const bh = h.oddsH || 2.10, bd = h.oddsD || 3.40, ba = h.oddsA || 3.60
      const ip = impliedProbs(bh, bd, ba)
      const mktH = ip.home, mktD = ip.draw, mktA = ip.away

      const probMap = { H: h.probH, D: h.probD, A: h.probA }
      const oddsMap = { H: bh, D: bd, A: ba }
      const mktMap = { H: mktH, D: mktD, A: mktA }
      const predProb = probMap[h.prediction] || 0
      const predOdds = oddsMap[h.prediction] || 1
      const predMkt = mktMap[h.prediction] || 0

      const ev = calcEV(predProb, predOdds)
      const edge = calcEdge(predProb, predMkt)
      const kelly = calcKelly(predProb, predOdds)
      const isVB = edge >= 10 && predOdds <= 5 && predProb >= 35

      return { ...h, ev, edge, kelly, isVB }
    })

    // Filter
    const filters = {
      all: () => true,
      vb: r => r.isVB,
      H: r => r.prediction === 'H',
      D: r => r.prediction === 'D',
      A: r => r.prediction === 'A',
    }
    data = data.filter(filters[filter] || filters.all)

    // Sort
    data.sort((a, b) => {
      const av = a[sort.col] ?? ''
      const bv = b[sort.col] ?? ''
      return sort.asc ? (av > bv ? 1 : -1) : (av < bv ? 1 : -1)
    })

    return data
  }, [history, filter, sort])

  const totalPages = Math.ceil(filtered.length / PER_PAGE)
  const pageData = filtered.slice((page - 1) * PER_PAGE, page * PER_PAGE)

  // KPIs
  const total = history.length
  const vbCount = history.filter(h => {
    const bh = h.oddsH || 2.10, bd = h.oddsD || 3.40, ba = h.oddsA || 3.60
    const ip = impliedProbs(bh, bd, ba)
    const probMap = { H: h.probH, D: h.probD, A: h.probA }
    const oddsMap = { H: bh, D: bd, A: ba }
    const mktMap = { H: ip.home, D: ip.draw, A: ip.away }
    const edge = calcEdge(probMap[h.prediction] || 0, mktMap[h.prediction] || 0)
    return edge >= 10 && (oddsMap[h.prediction] || 1) <= 5 && (probMap[h.prediction] || 0) >= 35
  }).length
  const avgConf = total > 0 ? (history.reduce((s, h) => s + (h.confidence || 0), 0) / total * 100).toFixed(0) : 0

  const handleSort = (col) => {
    setSort(prev => prev.col === col ? { col, asc: !prev.asc } : { col, asc: false })
    setPage(1)
  }

  const handleFilter = (f) => {
    setFilter(f)
    setPage(1)
  }

  const handleClear = () => {
    if (confirm('\u00BFBorrar todo el historial?')) {
      clearHistory()
      setHistory([])
    }
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
            ].map(([key, label]) => (
              <button key={key} className={`flt${filter === key ? ' on' : ''}`}
                onClick={() => handleFilter(key)}>
                {label}
              </button>
            ))}
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
                    <th onClick={() => handleSort('timestamp')}>Fecha</th>
                  </tr>
                </thead>
                <tbody>
                  {pageData.map(h => (
                    <tr key={h.id}>
                      <td style={{ color: 'var(--ink4)', fontWeight: 600, fontFamily: 'var(--mono)' }}>
                        J{h.jornada}
                      </td>
                      <td style={{ fontWeight: 600 }}>{h.home}</td>
                      <td style={{ color: 'var(--ink3)' }}>{h.away}</td>
                      <td>
                        <span className={`bgt bgt-${h.prediction}`}>
                          {PRED_LABELS[h.prediction]}
                        </span>
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
                          : <span style={{ color: 'var(--ink5)' }}>{'\u2014'}</span>
                        }
                      </td>
                      <td style={{ fontFamily: 'var(--mono)', color: h.kelly > 0 ? 'var(--amb)' : 'var(--ink5)' }}>
                        {h.kelly > 0 ? `${h.kelly}%` : '\u2014'}
                      </td>
                      <td style={{ fontSize: 10, color: 'var(--ink4)' }}>
                        {new Date(h.timestamp).toLocaleDateString()}
                      </td>
                    </tr>
                  ))}
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
