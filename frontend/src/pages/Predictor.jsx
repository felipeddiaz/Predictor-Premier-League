import { useState, useEffect, useCallback } from 'react'
import { api } from '../services/api'
import {
  teamInitials, impliedProbs, calcEdge, calcEV, calcKelly,
  entropyConfidence, recommendation, mapPrediction, saveToHistory,
} from '../utils/calc'
import Loader from '../components/Loader'

const FALLBACK_TEAMS = [
  'Arsenal','Aston Villa','Bournemouth','Brentford','Brighton',
  'Chelsea','Crystal Palace','Everton','Fulham','Ipswich',
  'Leicester','Liverpool','Man City','Man United','Newcastle',
  "Nott'm Forest",'Southampton','Tottenham','West Ham','Wolves',
]

export default function Predictor() {
  const [teams, setTeams] = useState(FALLBACK_TEAMS)
  const [matches, setMatches] = useState([])
  const [activeId, setActiveId] = useState(null)
  const [jornada, setJornada] = useState(30)
  const [loading, setLoading] = useState(false)
  const [counter, setCounter] = useState(0)

  useEffect(() => {
    api.teams()
      .then(data => { if (Array.isArray(data) && data.length) setTeams(data) })
      .catch(() => {})
  }, [])

  const addMatch = useCallback(() => {
    setCounter(c => {
      const id = c + 1
      const home = teams[id % teams.length]
      const away = teams[(id + 5) % teams.length]
      setMatches(prev => [...prev, { id, home, away, bh: '', bd: '', ba: '', result: null }])
      setActiveId(id)
      return id
    })
  }, [teams])

  const removeMatch = useCallback((id) => {
    setMatches(prev => prev.filter(m => m.id !== id))
    setActiveId(cur => cur === id ? null : cur)
  }, [])

  const updateField = useCallback((id, field, value) => {
    setMatches(prev => prev.map(m =>
      m.id === id ? { ...m, [field]: value, result: null } : m
    ))
  }, [])

  const predictMatch = useCallback(async (id) => {
    const m = matches.find(x => x.id === id)
    if (!m) return

    const bh = parseFloat(m.bh) || 2.10
    const bd = parseFloat(m.bd) || 3.40
    const ba = parseFloat(m.ba) || 3.60

    setLoading(true)
    try {
      let data
      if (m.bh && m.bd && m.ba) {
        data = await api.predictDetail(m.home, m.away, bh, bd, ba)
      } else {
        data = await api.predictSimple(m.home, m.away)
      }
      const result = mapPrediction(data, bh, bd, ba)

      setMatches(prev => prev.map(x => x.id === id ? { ...x, result } : x))

      // Save to history
      saveToHistory({
        jornada,
        home: m.home,
        away: m.away,
        prediction: result.prediction,
        probH: result.probH,
        probD: result.probD,
        probA: result.probA,
        oddsH: bh, oddsD: bd, oddsA: ba,
        confidence: result.confidence,
      })
    } catch (err) {
      alert(`Error: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }, [matches, jornada])

  const activeMatch = matches.find(m => m.id === activeId)

  return (
    <>
      <Loader visible={loading} text="Analizando partido\u2026" />
      <div className="pg-pred">
        {/* SIDEBAR */}
        <div className="sidebar">
          <div className="sb-top">
            <div className="sb-top-row">
              <span className="sb-heading">Partidos</span>
              <div className="jornada-ctrl">
                J<input type="number" value={jornada} min={1} max={38}
                  onChange={e => setJornada(+e.target.value)} />/ 38
              </div>
            </div>
            <button className="btn-add" onClick={addMatch}>+ Agregar partido</button>
          </div>
          <div className="sb-list">
            {!matches.length && (
              <div style={{ padding: 16, textAlign: 'center', color: 'var(--ink4)', fontSize: 12 }}>
                Sin partidos a\u00fan
              </div>
            )}
            {matches.map(m => (
              <MatchItem key={m.id} match={m} active={activeId === m.id}
                onClick={() => setActiveId(m.id)} />
            ))}
          </div>
        </div>

        {/* DETAIL PANEL */}
        <div className="detail-panel">
          {activeMatch ? (
            <MatchDetail
              match={activeMatch}
              teams={teams}
              jornada={jornada}
              onUpdate={updateField}
              onPredict={predictMatch}
              onRemove={removeMatch}
            />
          ) : (
            <div className="d-empty">
              <div className="d-empty-icon">{'\u26BD'}</div>
              <div className="d-empty-msg">Agrega un partido para comenzar</div>
            </div>
          )}
        </div>
      </div>
    </>
  )
}

/* ── Sidebar Match Item ── */
function MatchItem({ match: m, active, onClick }) {
  const predLabel = { H: '1', D: 'X', A: '2' }
  return (
    <div className={`mi${active ? ' on' : ''}`} onClick={onClick}>
      <div className="mi-teams">
        <div className="mi-team">
          <div className="mi-badge">{teamInitials(m.home)}</div>
          <span className="mi-name">{m.home}</span>
          {m.result && <span className={`bgt bgt-${m.result.prediction}`} style={{ fontSize: 8, padding: '1px 5px' }}>
            {predLabel[m.result.prediction]}
          </span>}
        </div>
        <div className="mi-team">
          <div className="mi-badge" style={{ opacity: .5 }}>{teamInitials(m.away)}</div>
          <span className="mi-name away">{m.away}</span>
        </div>
      </div>
      {(m.bh || m.bd || m.ba) && (
        <div className="mi-bottom">
          <div className="mi-odd">{m.bh || '\u2014'}</div>
          <div className="mi-odd">{m.bd || '\u2014'}</div>
          <div className="mi-odd">{m.ba || '\u2014'}</div>
          {m.result && (
            <span className={`mi-status ${m.result.source === 'api' ? 'predicted' : 'predicted'}`}>
              {'\u2713'}
            </span>
          )}
        </div>
      )}
    </div>
  )
}

/* ── Match Detail Panel ── */
function MatchDetail({ match: m, teams, jornada, onUpdate, onPredict, onRemove }) {
  const res = m.result
  const bh = parseFloat(m.bh) || 2.10
  const bd = parseFloat(m.bd) || 3.40
  const ba = parseFloat(m.ba) || 3.60

  return (
    <>
      {/* HEADER CARD */}
      <div className="w-header">
        <div className="wh-team">
          <div className="wh-crest">{teamInitials(m.home)}</div>
          <div>
            <label style={{ fontSize: 9, fontWeight: 600, color: 'var(--ink4)', display: 'block', marginBottom: 3 }}>LOCAL</label>
            <select style={{ fontSize: 12, fontWeight: 700, padding: '5px 8px' }}
              value={m.home} onChange={e => onUpdate(m.id, 'home', e.target.value)}>
              {teams.map(t => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
        </div>

        <div className="wh-center">
          <div className="wh-league">Premier League &middot; J{jornada}</div>
          <div className="wh-vs" style={{ fontSize: 16, fontWeight: 800, margin: '4px 0' }}>vs</div>
          <div className={`wh-status${res ? ' done' : ''}`}>
            {'\u25CF'} {res ? 'Predicho' : 'Pendiente'}
          </div>
          <div className="wh-odds" style={{ marginTop: 8 }}>
            {[['1', 'bh', '2.10'], ['X', 'bd', '3.40'], ['2', 'ba', '3.60']].map(([lbl, field, ph]) => (
              <div className="wh-odd-box" key={field}>
                <span className="wh-odd-lbl">{lbl}</span>
                <input type="number" className="wh-odd-inp" step=".01" min="1.01"
                  placeholder={ph} value={m[field]}
                  onChange={e => onUpdate(m.id, field, e.target.value)} />
              </div>
            ))}
          </div>
          <div className="wh-actions">
            <button className="btn-predict" onClick={() => onPredict(m.id)}>
              Predecir {'\u2192'}
            </button>
            <button className="btn-del" onClick={() => onRemove(m.id)}>
              {'\u2715'}
            </button>
          </div>
        </div>

        <div className="wh-team" style={{ alignItems: 'center' }}>
          <div className="wh-crest" style={{ opacity: .75 }}>{teamInitials(m.away)}</div>
          <div>
            <label style={{ fontSize: 9, fontWeight: 600, color: 'var(--ink4)', display: 'block', marginBottom: 3 }}>VISITANTE</label>
            <select style={{ fontSize: 12, fontWeight: 700, padding: '5px 8px' }}
              value={m.away} onChange={e => onUpdate(m.id, 'away', e.target.value)}>
              {teams.map(t => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
        </div>
      </div>

      {/* ANALYSIS WIDGETS (only if predicted) */}
      {res && <AnalysisWidgets result={res} match={m} bh={bh} bd={bd} ba={ba} />}
    </>
  )
}

/* ── Analysis Widgets ── */
function AnalysisWidgets({ result: res, match: m, bh, bd, ba }) {
  // Market calculations
  const mkt = res.mktH != null
    ? { h: res.mktH, d: res.mktD, a: res.mktA }
    : (() => { const ip = impliedProbs(bh, bd, ba); return { h: ip.home, d: ip.draw, a: ip.away } })()

  const rawH = 1/bh, rawD = 1/bd, rawA = 1/ba
  const ov = rawH + rawD + rawA
  const vig = +((ov - 1) * 100).toFixed(1)

  const eH = calcEdge(res.probH, mkt.h)
  const eD = calcEdge(res.probD, mkt.d)
  const eA = calcEdge(res.probA, mkt.a)

  const evH = calcEV(res.probH, bh)
  const evD = calcEV(res.probD, bd)
  const evA = calcEV(res.probA, ba)

  const kH = calcKelly(res.probH, bh)
  const kD = calcKelly(res.probD, bd)
  const kA = calcKelly(res.probA, ba)

  const rH = recommendation(eH, evH, kH, bh)
  const rD = recommendation(eD, evD, kD, bd)
  const rA = recommendation(eA, evA, kA, ba)

  const conf = entropyConfidence(res.probH, res.probD, res.probA)
  const maxEdge = Math.max(...[eH, eD, eA].map(e => Math.abs(e)), 10)

  const ranked = [
    { k: 'H', name: 'Local', ev: evH, edge: eH, stake: kH, rec: rH, odds: bh, model: res.probH, mkt: mkt.h },
    { k: 'D', name: 'Empate', ev: evD, edge: eD, stake: kD, rec: rD, odds: bd, model: res.probD, mkt: mkt.d },
    { k: 'A', name: 'Visitante', ev: evA, edge: eA, stake: kA, rec: rA, odds: ba, model: res.probA, mkt: mkt.a },
  ].sort((a, b) => b.ev - a.ev)

  const probData = [
    { k: 'H', label: 'Local', prob: res.probH, mktP: mkt.h, edge: eH },
    { k: 'D', label: 'Empate', prob: res.probD, mktP: mkt.d, edge: eD },
    { k: 'A', label: 'Visitante', prob: res.probA, mktP: mkt.a, edge: eA },
  ]
  const bestProb = Math.max(res.probH, res.probD, res.probA)

  const edgeBar = (lbl, edge) => {
    const e = parseFloat(edge)
    const pos = e >= 0
    const w = Math.min(100, Math.abs(e) / maxEdge * 100).toFixed(0)
    return (
      <div className="edge-bar-row">
        <div className="eb-lbl">{lbl}</div>
        <div className="eb-track">
          <div className={`eb-fill ${pos ? 'eb-pos' : 'eb-neg'}`} style={{ width: `${w}%` }} />
        </div>
        <div className={`eb-val ${pos ? 'p' : 'n'}`}>{pos ? '+' : ''}{edge}%</div>
      </div>
    )
  }

  return (
    <>
      {/* ROW 2: PROBABILITIES + MARKET */}
      <div className="w-row2">
        <div className="wcard">
          <div className="wc-head">
            <span className="wc-title">Probabilidades</span>
            <span className="wc-meta" style={{ color: 'var(--grn)', fontWeight: 700 }}>
              {'\u2713'} Modelo real
            </span>
          </div>
          <div className="wc-body">
            <div className="prob-cols">
              {probData.map(({ k, label, prob, mktP, edge: e }) => {
                const eNum = parseFloat(e)
                return (
                  <div key={k} className={`prob-col${prob === bestProb ? ' best' : ''}`}>
                    <div className="pc-lbl">{label}</div>
                    <div className={`pc-val ${k}`}>{prob}%</div>
                    <div className="pc-sub">Mkt {mktP}%</div>
                    <div style={{ marginTop: 4 }}>
                      <span style={{
                        fontSize: 9, fontWeight: 700, fontFamily: 'var(--mono)',
                        color: eNum >= 0 ? 'var(--grn)' : 'var(--red)',
                        background: eNum >= 0 ? 'var(--grn-l)' : 'var(--red-l)',
                        border: `1px solid ${eNum >= 0 ? 'var(--grn-b)' : 'var(--red-b)'}`,
                        padding: '1px 5px', borderRadius: 3, display: 'inline-block',
                      }}>
                        {eNum >= 0 ? '+' : ''}{e}%
                      </span>
                    </div>
                  </div>
                )
              })}
            </div>
            <div className="prob-bar" style={{
              gridTemplateColumns: `${res.probH}fr ${res.probD}fr ${res.probA}fr`,
              marginTop: 10,
            }}>
              <div className="pb-H" /><div className="pb-D" /><div className="pb-A" />
            </div>
          </div>
        </div>

        <div className="wcard">
          <div className="wc-head"><span className="wc-title">Mercado y tendencias</span></div>
          <div className="wc-body">
            <div className="mkt-strip">
              <div className="ms-cell">
                <div className="ms-lbl">Overround</div>
                <div className="ms-val" style={{ color: vig > 4 ? 'var(--red)' : 'var(--grn)' }}>
                  {(vig + 100).toFixed(1)}%
                </div>
              </div>
              <div className="ms-cell">
                <div className="ms-lbl">Vig</div>
                <div className="ms-val" style={{ color: 'var(--amb)' }}>{vig}%</div>
              </div>
              <div className="ms-cell">
                <div className="ms-lbl">Confianza</div>
                <div className="ms-val" style={{ color: conf.color }}>{conf.label}</div>
              </div>
            </div>
            <div className="edge-bar-wrap">
              <div style={{ fontSize: 9, fontWeight: 700, color: 'var(--ink4)', textTransform: 'uppercase', letterSpacing: '.5px', marginBottom: 5 }}>
                Edge modelo vs mercado
              </div>
              {edgeBar('1', eH)}
              {edgeBar('X', eD)}
              {edgeBar('2', eA)}
            </div>
          </div>
        </div>
      </div>

      {/* ROW 3: FORM + RECOMMENDATION */}
      <div className="w-row3">
        <div className="wcard">
          <div className="wc-head">
            <span className="wc-title">Forma reciente</span>
            <span className="wc-meta">{'\u00DAltimos'} 5 partidos</span>
          </div>
          <div className="wc-body">
            <div className="forma-section">
              {[{ name: m.home, form: res.formHome }, { name: m.away, form: res.formAway }].map(({ name, form }, i) => (
                <div className="forma-team" key={i} style={i > 0 ? { marginTop: 6 } : {}}>
                  <span className="ft-name" style={{ fontSize: 11 }}>{name.split(' ')[0]}</span>
                  <div className="ft-dots">
                    {form.length ? form.map((r, j) => (
                      <div key={j} className={`ft-dot ${r}`}>{r}</div>
                    )) : (
                      <span style={{ fontSize: 10, color: 'var(--ink4)' }}>Sin datos</span>
                    )}
                  </div>
                </div>
              ))}
            </div>

            {/* Binary markets */}
            {(res.over25 != null || res.over35cards != null || res.over95corners != null) && (
              <div style={{ marginTop: 10, borderTop: '1px solid var(--border)', paddingTop: 8 }}>
                <div style={{ fontSize: 9, fontWeight: 700, color: 'var(--ink4)', textTransform: 'uppercase', letterSpacing: '.5px', marginBottom: 6 }}>
                  Mercados binarios
                </div>
                <div className="stats-grid">
                  {res.over25 != null && (
                    <div className="stat-block">
                      <div className="sb-lbl">Over 2.5</div>
                      <div className="sb-home">{res.over25}%</div>
                    </div>
                  )}
                  {res.over35cards != null && (
                    <div className="stat-block">
                      <div className="sb-lbl">+3.5 Tarj.</div>
                      <div className="sb-home">{res.over35cards}%</div>
                    </div>
                  )}
                  {res.over95corners != null && (
                    <div className="stat-block">
                      <div className="sb-lbl">+9.5 Corn.</div>
                      <div className="sb-home">{res.over95corners}%</div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="wcard">
          <div className="wc-head">
            <span className="wc-title">Recomendaci{'\u00F3'}n</span>
            <span className="wc-meta">Sistema automatizado</span>
          </div>
          <div className="wc-body">
            <div className="rec-rows">
              {ranked.map((o) => (
                <div key={o.k} className={`rec-row${o.rec.cls ? ' ' + o.rec.cls : ''}`}>
                  <span className="rr-icon">{o.rec.icon}</span>
                  <span className="rr-name">
                    {o.name}{' '}
                    <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--ink4)' }}>
                      @ {o.odds}
                    </span>
                  </span>
                  <span className={`rr-ev ${o.ev >= 0 ? 'p' : 'n'}`}>
                    {o.ev >= 0 ? '+' : ''}{o.ev}
                  </span>
                  <span className={`rr-stake ${o.rec.stake > 0 ? 'vb' : 'none'}`}>
                    {o.rec.stake > 0 ? `${o.rec.stake}% bk` : '\u2014'}
                  </span>
                </div>
              ))}
            </div>

            <div style={{ marginTop: 10, borderTop: '1px solid var(--border)', paddingTop: 8 }}>
              <div style={{ fontSize: 9, fontWeight: 700, color: 'var(--ink4)', textTransform: 'uppercase', letterSpacing: '.5px', marginBottom: 6 }}>
                Value ranking
              </div>
              <table className="rank-mini">
                <thead>
                  <tr><th>#</th><th>Resultado</th><th>Edge</th><th>EV</th><th>Kelly</th></tr>
                </thead>
                <tbody>
                  {ranked.map((o, i) => (
                    <tr key={o.k}>
                      <td style={{ color: 'var(--ink4)', fontFamily: 'var(--mono)' }}>{i + 1}</td>
                      <td><span className={`bgt bgt-${o.k}`}>{o.name}</span></td>
                      <td style={{ fontFamily: 'var(--mono)', color: o.edge >= 0 ? 'var(--grn)' : 'var(--red)' }}>
                        {o.edge >= 0 ? '+' : ''}{o.edge}%
                      </td>
                      <td style={{ fontFamily: 'var(--mono)', color: o.ev >= 0 ? 'var(--grn)' : 'var(--red)' }}>
                        {o.ev >= 0 ? '+' : ''}{o.ev}
                      </td>
                      <td style={{ fontFamily: 'var(--mono)', color: 'var(--amb)' }}>
                        {o.rec.stake > 0 ? `${o.rec.stake}%` : '\u2014'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
