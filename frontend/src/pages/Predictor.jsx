import { useState, useEffect, useCallback } from 'react'
import { api } from '../services/api'
import {
  impliedProbs, calcEdge, calcEV, calcKelly,
  entropyConfidence, recommendation, parseForm, saveToHistory,
} from '../utils/calc'
import { TeamBadge } from '../utils/teamLogos.jsx'
import Loader from '../components/Loader'

export default function Predictor() {
  const [matches, setMatches] = useState([])
  const [activeIdx, setActiveIdx] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [meta, setMeta] = useState(null)

  useEffect(() => { loadMatches() }, [])

  const loadMatches = async (forceRefresh = false) => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.matches(forceRefresh)
      const list = (data.matches || []).map((m, i) => ({
        ...m,
        id: i,
        pred: m.prediction ? {
          probH: +(m.prediction.prob_home * 100).toFixed(1),
          probD: +(m.prediction.prob_draw * 100).toFixed(1),
          probA: +(m.prediction.prob_away * 100).toFixed(1),
          mktH: m.prediction.market_prob_home ? +(m.prediction.market_prob_home * 100).toFixed(1) : null,
          mktD: m.prediction.market_prob_draw ? +(m.prediction.market_prob_draw * 100).toFixed(1) : null,
          mktA: m.prediction.market_prob_away ? +(m.prediction.market_prob_away * 100).toFixed(1) : null,
          prediction: { 'Local': 'H', 'Empate': 'D', 'Visitante': 'A' }[m.prediction.result] || 'H',
          confidence: m.prediction.confidence,
          formHome: parseForm(m.prediction.form_home),
          formAway: parseForm(m.prediction.form_away),
          over25: m.prediction.over25_prob ? +(m.prediction.over25_prob * 100).toFixed(1) : null,
          over35cards: m.prediction.over35_cards_prob ? +(m.prediction.over35_cards_prob * 100).toFixed(1) : null,
          over95corners: m.prediction.over95_corners_prob ? +(m.prediction.over95_corners_prob * 100).toFixed(1) : null,
        } : null,
      }))

      setMatches(list)
      setMeta(data.meta || {})
      if (list.length) setActiveIdx(0)

      list.forEach(m => {
        if (m.pred) {
          saveToHistory({
            home: m.home, away: m.away,
            prediction: m.pred.prediction,
            probH: m.pred.probH, probD: m.pred.probD, probA: m.pred.probA,
            oddsH: m.odds?.home, oddsD: m.odds?.draw, oddsA: m.odds?.away,
            confidence: m.pred.confidence,
            jornada: 0,
          })
        }
      })
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const activeMatch = matches[activeIdx]

  return (
    <>
      <Loader visible={loading} text="Cargando partidos..." />
      <div className="pg-pred">
        {/* SIDEBAR */}
        <div className="sidebar">
          <div className="sb-top">
            <div className="sb-top-row">
              <span className="sb-heading">Partidos</span>
              <span style={{ fontSize: 10, color: 'var(--color-text-subtle)' }}>{matches.length} partidos</span>
            </div>
            {meta?.last_updated && (
              <div style={{ fontSize: 10, color: 'var(--color-text-subtle)', marginBottom: 6 }}>
                Actualizado: {new Date(meta.last_updated).toLocaleString()}
              </div>
            )}
            <button className="btn-add" onClick={() => loadMatches(true)}
              style={{ borderStyle: 'solid', background: 'var(--color-bg-base)' }}>
              Actualizar cuotas
            </button>
          </div>
          <div className="sb-list">
            {!matches.length && !loading && (
              <div style={{ padding: 16, textAlign: 'center', color: 'var(--color-text-subtle)', fontSize: 12 }}>
                {error || 'Sin partidos disponibles'}
              </div>
            )}
            {matches.map((m, i) => (
              <SidebarItem key={i} match={m} active={activeIdx === i} onClick={() => setActiveIdx(i)} />
            ))}
          </div>
        </div>

        {/* DETAIL PANEL */}
        <div className="detail-panel">
          {error && !matches.length && (
            <div className="d-empty">
              <div className="d-empty-icon">{'\u26A0\uFE0F'}</div>
              <div className="d-empty-msg">{error}</div>
              <button className="btn-predict" onClick={() => loadMatches()} style={{ marginTop: 10 }}>
                Reintentar
              </button>
            </div>
          )}
          {activeMatch ? (
            <MatchDetail match={activeMatch} />
          ) : !loading && !error && (
            <div className="d-empty">
              <div className="d-empty-icon">{'\u26BD'}</div>
              <div className="d-empty-msg">Selecciona un partido</div>
            </div>
          )}
        </div>
      </div>
    </>
  )
}

/* ─────────────────────────────────────────
   SIDEBAR ITEM
───────────────────────────────────────── */
function SidebarItem({ match: m, active, onClick }) {
  const pred = m.pred
  const predLabel = { H: '1', D: 'X', A: '2' }
  return (
    <div className={`mi${active ? ' on' : ''}`} onClick={onClick}>
      <div className="mi-teams">
        <div className="mi-team">
          <TeamBadge name={m.home} size={22} />
          <span className="mi-name">{m.home}</span>
          {pred && <span className={`bgt bgt-${pred.prediction}`} style={{ fontSize: 8, padding: '1px 5px' }}>
            {predLabel[pred.prediction]}
          </span>}
        </div>
        <div className="mi-team">
          <TeamBadge name={m.away} size={22} style={{ opacity: .6 }} />
          <span className="mi-name away">{m.away}</span>
        </div>
      </div>
      {m.odds && (
        <div className="mi-bottom">
          <div className="mi-odd">{m.odds.home}</div>
          <div className="mi-odd">{m.odds.draw}</div>
          <div className="mi-odd">{m.odds.away}</div>
          {m.odds_changed && <span className="mi-status vb">UPD</span>}
          {pred && !m.odds_changed && <span className="mi-status predicted">{'\u2713'}</span>}
        </div>
      )}
    </div>
  )
}

/* ─────────────────────────────────────────
   MATCH DETAIL — orchestrates 3 blocks
───────────────────────────────────────── */
function MatchDetail({ match: m }) {
  const pred = m.pred
  const odds = m.odds || {}
  const bh = odds.home || 2.80
  const bd = odds.draw || 3.40
  const ba = odds.away || 2.80
  const [statsTeam, setStatsTeam] = useState(null)

  return (
    <>
      {/* BLOCK 1 */}
      <MatchHeaderV2 match={m} pred={pred} bh={bh} bd={bd} ba={ba} onTeamClick={setStatsTeam} />

      {/* BLOCK 2 */}
      {pred && (
        <div className="v2-analysis">
          <ProbCard pred={pred} bh={bh} bd={bd} ba={ba} />
          <TabbedPanel pred={pred} bh={bh} bd={bd} ba={ba} />
        </div>
      )}

      {/* BLOCK 3 */}
      {pred && (pred.over25 != null || pred.over35cards != null || pred.over95corners != null) && (
        <BinaryRow pred={pred} />
      )}

      {/* H2H */}
      <H2HWidget home={m.home} away={m.away} />

      {!pred && (
        <div className="wcard">
          <div className="wc-body" style={{ textAlign: 'center', padding: 20, color: 'var(--color-text-subtle)' }}>
            No se pudo generar predicci\u00F3n para este partido
          </div>
        </div>
      )}

      {statsTeam && (
        <TeamStatsModal name={statsTeam.name} isLocal={statsTeam.isLocal} onClose={() => setStatsTeam(null)} />
      )}
    </>
  )
}

/* ─────────────────────────────────────────
   BLOCK 1 — MATCH HEADER V2
───────────────────────────────────────── */
function MatchHeaderV2({ match: m, pred, bh, bd, ba, onTeamClick }) {
  const minOdds = Math.min(bh, bd, ba)
  const favKey = bh === minOdds ? 'H' : bd === minOdds ? 'D' : 'A'
  const predLabels = { H: 'Local', D: 'Empate', A: 'Visitante' }

  return (
    <div className="v2-header">
      {pred && (
        <div className="v2-pred-badge">
          <span className="v2-pulse" />
          {predLabels[pred.prediction]}
        </div>
      )}
      <div className="v2-header-grid">
        <TeamCol
          name={m.home} role="Local" form={pred?.formHome}
          onClick={() => onTeamClick({ name: m.home, isLocal: true })}
        />
        <div className="v2-center">
          <span className="v2-vs">VS</span>
          {m.commence_time && (
            <div className="v2-matchtime">
              {new Date(m.commence_time).toLocaleDateString()}{' '}
              {new Date(m.commence_time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </div>
          )}
          <div className="v2-odds-row">
            {[['1', bh, 'H'], ['X', bd, 'D'], ['2', ba, 'A']].map(([lbl, val, key]) => (
              <div key={key} className={`v2-odds-pill${key === favKey ? ' fav' : ''}`}>
                <span className="v2-odds-lbl">{lbl}</span>
                <span className="v2-odds-val">{val}</span>
              </div>
            ))}
          </div>
          {m.odds?.n_bookmakers > 0 && (
            <div className="v2-bk-note">Media {m.odds.n_bookmakers} casas</div>
          )}
          {m.odds_changed && m.odds_previous && (
            <div style={{ fontSize: 9, color: 'var(--color-text-warning)', fontWeight: 700, marginTop: 2 }}>
              Cuotas actualizadas
            </div>
          )}
        </div>
        <TeamCol
          name={m.away} role="Visitante" form={pred?.formAway}
          onClick={() => onTeamClick({ name: m.away, isLocal: false })}
        />
      </div>
    </div>
  )
}

function TeamCol({ name, role, form, onClick }) {
  return (
    <div className="v2-team-col" onClick={onClick}>
      <TeamBadge name={name} size={48} style={{ borderRadius: '50%' }} />
      <div className="v2-team-name">{name}</div>
      <div className="v2-team-role">{role}</div>
      {form && form.length > 0 && (
        <div className="v2-form-pills">
          {form.slice(-5).map((r, i) => (
            <span key={i} className={`v2-fp v2-fp-${r}`}>{r}</span>
          ))}
        </div>
      )}
    </div>
  )
}

/* ─────────────────────────────────────────
   BLOCK 2 LEFT — PROBABILITIES CARD
───────────────────────────────────────── */
function ProbCard({ pred, bh, bd, ba }) {
  const mkt = pred.mktH != null
    ? { h: pred.mktH, d: pred.mktD, a: pred.mktA }
    : (() => { const ip = impliedProbs(bh, bd, ba); return { h: ip.home, d: ip.draw, a: ip.away } })()

  const rawH = 1 / bh, rawD = 1 / bd, rawA = 1 / ba
  const ov = rawH + rawD + rawA
  const vig = +((ov - 1) * 100).toFixed(1)
  const overround = +(ov * 100).toFixed(1)

  const eH = calcEdge(pred.probH, mkt.h)
  const eD = calcEdge(pred.probD, mkt.d)
  const eA = calcEdge(pred.probA, mkt.a)
  const conf = entropyConfidence(pred.probH, pred.probD, pred.probA)
  const maxEdge = Math.max(Math.abs(eH), Math.abs(eD), Math.abs(eA), 10)

  const bestProb = Math.max(pred.probH, pred.probD, pred.probA)
  const items = [
    { k: 'H', label: 'Local',     prob: pred.probH, mktP: mkt.h, edge: eH },
    { k: 'D', label: 'Empate',    prob: pred.probD, mktP: mkt.d, edge: eD },
    { k: 'A', label: 'Visitante', prob: pred.probA, mktP: mkt.a, edge: eA },
  ]

  return (
    <div className="wcard">
      <div className="wc-head">
        <span className="wc-title">Probabilidades</span>
        <span className="v2-model-badge">{'\u2713'} XGBoost</span>
      </div>
      <div className="wc-body">
        <div className="v2-prob-cols">
          {items.map(({ k, label, prob, mktP, edge }) => {
            const eNum = parseFloat(edge)
            return (
              <div key={k} className={`v2-prob-item${prob === bestProb ? ' best' : ''}${eNum >= 10 ? ' vb-edge' : ''}`}>
                <div className="v2-pi-lbl">{label}</div>
                <div className={`v2-pi-pct v2-pct-${k}`}>{prob}%</div>
                <div className="v2-pi-bar">
                  <div className={`v2-pi-fill v2-fill-${k}`} style={{ width: `${prob}%` }} />
                </div>
                <div className="v2-pi-mkt">{mktP}% mkt</div>
                <div className={`v2-pi-edge ${eNum >= 0 ? 'pos' : 'neg'}`}>
                  {eNum >= 0 ? '+' : ''}{edge}%
                </div>
              </div>
            )
          })}
        </div>

        <div className="v2-prob-footer">
          {/* Edge bars */}
          <div className="v2-edge-section">
            <div className="v2-section-lbl">Edge vs mercado</div>
            {items.map(({ k, label, edge }) => {
              const e = parseFloat(edge)
              const w = Math.min(100, Math.abs(e) / maxEdge * 100)
              return (
                <div key={k} className="v2-edge-row">
                  <span className="v2-edge-lbl">{label[0]}</span>
                  <div className="v2-edge-track">
                    <div className={`v2-edge-fill ${e >= 0 ? 'pos' : 'neg'}`} style={{ width: `${w}%` }} />
                  </div>
                  <span className={`v2-edge-num ${e >= 0 ? 'pos' : 'neg'}`}>
                    {e >= 0 ? '+' : ''}{edge}%
                  </span>
                </div>
              )
            })}
          </div>

          <div className="v2-vdivider" />

          {/* Market metrics */}
          <div className="v2-mkt-section">
            <div className="v2-mm-lbl">Overround</div>
            <div className="v2-mm-val warn">{overround}%</div>
            <div className="v2-mm-lbl">VIG</div>
            <div className="v2-mm-val warn">{vig}%</div>
            <div className={`v2-conf-banner${conf.value >= 40 ? ' ok' : ''}`}>
              {conf.value < 40 ? '\u26A0 Confianza baja' : `Confianza: ${conf.label}`}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

/* ─────────────────────────────────────────
   BLOCK 2 RIGHT — TABBED PANEL
───────────────────────────────────────── */
function TabbedPanel({ pred, bh, bd, ba }) {
  const [tab, setTab] = useState('rec')

  const mkt = pred.mktH != null
    ? { h: pred.mktH, d: pred.mktD, a: pred.mktA }
    : (() => { const ip = impliedProbs(bh, bd, ba); return { h: ip.home, d: ip.draw, a: ip.away } })()

  const eH = calcEdge(pred.probH, mkt.h)
  const eD = calcEdge(pred.probD, mkt.d)
  const eA = calcEdge(pred.probA, mkt.a)
  const evH = calcEV(pred.probH, bh)
  const evD = calcEV(pred.probD, bd)
  const evA = calcEV(pred.probA, ba)
  const kH = calcKelly(pred.probH, bh)
  const kD = calcKelly(pred.probD, bd)
  const kA = calcKelly(pred.probA, ba)
  const rH = recommendation(eH, evH, kH, bh)
  const rD = recommendation(eD, evD, kD, bd)
  const rA = recommendation(eA, evA, kA, ba)

  const ranked = [
    { k: 'H', name: 'Local',     ev: evH, edge: eH, prob: pred.probH, rec: rH, odds: bh },
    { k: 'D', name: 'Empate',    ev: evD, edge: eD, prob: pred.probD, rec: rD, odds: bd },
    { k: 'A', name: 'Visitante', ev: evA, edge: eA, prob: pred.probA, rec: rA, odds: ba },
  ].sort((a, b) => b.ev - a.ev)

  return (
    <div className="wcard v2-tabpanel">
      <div className="v2-tab-switch">
        <button className={`v2-tab-btn${tab === 'rec' ? ' on' : ''}`} onClick={() => setTab('rec')}>
          Recomendaci\u00F3n
        </button>
        <button className={`v2-tab-btn${tab === 'vb' ? ' on' : ''}`} onClick={() => setTab('vb')}>
          Value betting
        </button>
      </div>
      <div className="v2-tab-sub">
        {tab === 'vb' ? 'Modelo XGBoost \u00B7 Kelly' : 'Sistema automatizado'}
      </div>
      <div key={tab} className="v2-tab-body">
        {tab === 'rec'
          ? <RecTab ranked={ranked} />
          : <VBTab ranked={ranked} />
        }
      </div>
    </div>
  )
}

/* ── Rec Tab ── */
function RecTab({ ranked }) {
  return (
    <div>
      <div className="v2-rec-rows">
        {ranked.map((o, i) => {
          const isDanger = o.ev < -0.05
          const isDim = i === 1 && !isDanger
          return (
            <div key={o.k} className={`v2-rec-row${isDanger ? ' danger' : ''}${isDim ? ' dim' : ''}`}>
              <span className="v2-rec-icon">{isDanger ? '\u2715' : '\u26A0'}</span>
              <div className="v2-rec-info">
                <span className={`bgt bgt-${o.k}`}>{o.name}</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--color-text-subtle)' }}>
                  @ {o.odds}
                </span>
              </div>
              <span className={`v2-rec-ev ${isDanger ? 'dng' : o.ev >= 0 ? 'pos' : 'neg'}`}>
                {o.ev >= 0 ? '+' : ''}{o.ev}
              </span>
              {o.rec.stake > 0 && <span className="v2-rec-kelly">{o.rec.stake}%</span>}
            </div>
          )
        })}
      </div>

      <div style={{ borderTop: '0.5px solid var(--color-border-base)', paddingTop: 8 }}>
        <div className="v2-section-lbl" style={{ marginBottom: 6 }}>Value ranking</div>
        <table className="rank-mini">
          <thead>
            <tr><th>#</th><th>Resultado</th><th>Edge</th><th>EV</th><th>Kelly</th></tr>
          </thead>
          <tbody>
            {ranked.map((o, i) => (
              <tr key={o.k}>
                <td style={{ color: 'var(--color-text-subtle)', fontFamily: 'var(--font-mono)' }}>{i + 1}</td>
                <td><span className={`bgt bgt-${o.k}`}>{o.name}</span></td>
                <td style={{ fontFamily: 'var(--font-mono)', color: o.edge >= 0 ? 'var(--color-text-success)' : 'var(--color-text-danger)' }}>
                  {o.edge >= 0 ? '+' : ''}{o.edge}%
                </td>
                <td style={{ fontFamily: 'var(--font-mono)', color: o.ev >= 0 ? 'var(--color-text-success)' : 'var(--color-text-danger)' }}>
                  {o.ev >= 0 ? '+' : ''}{o.ev}
                </td>
                <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-text-warning)' }}>
                  {o.rec.stake > 0 ? `${o.rec.stake}%` : '\u2014'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

/* ── VB Tab ── */
function VBTab({ ranked }) {
  const bestVB = [...ranked].find(o => o.ev > 0)

  if (!bestVB) {
    return (
      <div className="v2-no-vb">
        Sin apuestas con valor positivo detectadas para este partido
      </div>
    )
  }

  const halfKelly = +(bestVB.rec.stake / 2).toFixed(2)
  const valueRatio = Math.min(100, Math.max(0, bestVB.edge) / 25 * 100)

  return (
    <div>
      <div className="v2-vb-detected">
        <div className="v2-vb-head">
          <span className="v2-vb-circle">{'\u2713'}</span>
          <span className="v2-vb-label">Apuesta con valor detectada</span>
          <span className="v2-vb-evbadge">+EV</span>
        </div>
        <div className="v2-vb-subcard">
          <div className="v2-grid4">
            <div className="v2-cell">
              <div className="v2-cell-lbl">Selecci\u00F3n</div>
              <div className="v2-cell-val"><span className={`bgt bgt-${bestVB.k}`}>{bestVB.name}</span></div>
            </div>
            <div className="v2-cell">
              <div className="v2-cell-lbl">Cuota</div>
              <div className="v2-cell-val" style={{ fontFamily: 'var(--font-mono)' }}>{bestVB.odds}</div>
            </div>
            <div className="v2-cell">
              <div className="v2-cell-lbl">Prob. justa</div>
              <div className="v2-cell-val" style={{ fontFamily: 'var(--font-mono)' }}>{bestVB.prob}%</div>
            </div>
            <div className="v2-cell">
              <div className="v2-cell-lbl">EV</div>
              <div className="v2-cell-val" style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-text-success)' }}>
                +{bestVB.ev}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="v2-grid2x2">
        <div className="v2-vb-metric">
          <div className="v2-vm-lbl">Kelly criterion</div>
          <div className="v2-vm-val">{bestVB.rec.stake}%</div>
        </div>
        <div className="v2-vb-metric">
          <div className="v2-vm-lbl">\u00BD Kelly</div>
          <div className="v2-vm-val">{halfKelly}%</div>
        </div>
        <div className="v2-vb-metric">
          <div className="v2-vm-lbl">Raz\u00F3n de valor</div>
          <div className="v2-val-track" style={{ marginTop: 8 }}>
            <div className="v2-val-fill" style={{ width: `${valueRatio}%` }} />
          </div>
          <div style={{ fontSize: 9, color: 'var(--color-text-subtle)', marginTop: 4, fontFamily: 'var(--font-mono)' }}>
            {bestVB.edge >= 0 ? '+' : ''}{bestVB.edge}% edge
          </div>
        </div>
        <div className="v2-vb-warn">
          <span>{'\u26A0'}</span>
          <span>Gestiona el riesgo. Nunca apuestes m\u00E1s de lo que puedes permitirte perder.</span>
        </div>
      </div>
    </div>
  )
}

/* ─────────────────────────────────────────
   BLOCK 3 — BINARY MARKETS
───────────────────────────────────────── */
function BinaryRow({ pred }) {
  const markets = [
    { label: 'Over 2.5',       value: pred.over25,       threshold: 55 },
    { label: '+3.5 Tarjetas',  value: pred.over35cards,  threshold: 55 },
    { label: '+9.5 C\u00F3rners', value: pred.over95corners, threshold: 55 },
  ].filter(m => m.value != null)

  if (!markets.length) return null

  return (
    <div className="wcard">
      <div className="wc-head">
        <span className="wc-title">Mercados binarios</span>
        <span className="wc-meta">Umbrales del modelo</span>
      </div>
      <div className="wc-body">
        <div className="v2-binary-grid">
          {markets.map(({ label, value, threshold }) => (
            <div key={label} className="v2-bin-card">
              <div className="v2-bin-lbl">{label}</div>
              <div className="v2-bin-pct">{value}%</div>
              <div className={`v2-bin-icon ${value >= threshold ? 'active' : 'idle'}`}>
                {value >= threshold ? '\u2713' : '\u00B7'}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

/* ─────────────────────────────────────────
   TEAM STATS MODAL
───────────────────────────────────────── */
function TeamStatsModal({ name, isLocal, onClose }) {
  const [tab, setTab] = useState(isLocal ? 'local' : 'visitante')
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(false)

  const loadStats = useCallback(async (asLocal) => {
    setLoading(true)
    try {
      const data = await api.teamStats(name, asLocal)
      setStats(data.stats || {})
    } catch {
      setStats({})
    } finally {
      setLoading(false)
    }
  }, [name])

  useEffect(() => { loadStats(tab === 'local') }, [tab, loadStats])

  const fmt = (v, dec = 2) => v != null ? (+v).toFixed(dec) : '\u2014'

  const ATTACK = [
    ['Goles prom.', 'AvgGoals', 1], ['xG prom.', 'xG_Avg', 2],
    ['xG global', 'xG_Global', 2], ['Disparos a puerta', 'AvgShotsTarget', 1],
    ['Goles (ult.5)', 'GoalsFor5', 1], ['SoR (ult.5)', 'SoR5', 2],
  ]
  const DEFENSE = [
    ['xGA prom.', 'xGA_Avg', 2], ['xGA global', 'xGA_Global', 2],
    ['Goles en contra (ult.5)', 'GoalsAgainst5', 1],
  ]
  const FORM = [
    ['Posici\u00F3n', 'Position', 0], ['Puntos', 'Points', 0],
    ['Racha', 'Streak', 0], ['Win Rate (ult.5)', 'WinRate5', 2],
    ['Pts (ult.5)', 'Pts5', 1], ['Momentum', 'Form_Momentum', 2],
  ]

  return (
    <div className="ts-overlay" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="ts-panel">
        <div className="ts-head">
          <TeamBadge name={name} size={28} />
          <span className="ts-title">{name}</span>
          <button className="ts-close" onClick={onClose}>{'\u00D7'}</button>
        </div>
        <div className="ts-tabs">
          {[['local', 'Como Local'], ['visitante', 'Como Visitante']].map(([key, lbl]) => (
            <button key={key} className={`ts-tab${tab === key ? ' on' : ''}`} onClick={() => setTab(key)}>
              {lbl}
            </button>
          ))}
        </div>
        <div className="ts-body">
          {loading && <div style={{ color: 'var(--color-text-subtle)', textAlign: 'center', padding: 20 }}>Cargando...</div>}
          {!loading && stats && (
            <>
              <div className="ts-group">
                <div className="ts-group-lbl">Forma reciente</div>
                <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
                  {[['W', stats.Form_W, 'var(--color-text-success)'], ['D', stats.Form_D, 'var(--color-text-warning)'], ['L', stats.Form_L, 'var(--color-text-danger)']].map(([l, v, c]) => (
                    <div key={l} style={{ flex: 1, background: 'var(--color-bg-base)', border: '0.5px solid var(--color-border-base)', borderRadius: 8, padding: '8px 0', textAlign: 'center' }}>
                      <div style={{ fontSize: 18, fontWeight: 800, fontFamily: 'var(--font-mono)', color: c }}>{fmt(v, 1)}</div>
                      <div style={{ fontSize: 9, color: 'var(--color-text-subtle)', fontWeight: 600, marginTop: 2 }}>{l === 'W' ? 'Victorias' : l === 'D' ? 'Empates' : 'Derrotas'}</div>
                    </div>
                  ))}
                </div>
              </div>
              {[['Ataque', ATTACK, 'var(--color-text-success)'], ['Defensa', DEFENSE, 'var(--color-text-danger)'], ['Tabla y tendencia', FORM, 'var(--color-text-base)']].map(([title, fields, color]) => (
                <div key={title} className="ts-group">
                  <div className="ts-group-lbl">{title}</div>
                  <div className="ts-stats-grid">
                    {fields.map(([lbl, key, dec]) => stats[key] != null && (
                      <div key={key} className="ts-stat">
                        <div className="ts-stat-lbl">{lbl}</div>
                        <div className="ts-stat-val" style={{ color, fontSize: 16 }}>{fmt(stats[key], dec)}</div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

/* ─────────────────────────────────────────
   H2H WIDGET
───────────────────────────────────────── */
function H2HWidget({ home, away }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    setData(null)
    api.h2h(home, away)
      .then(d => setData(d))
      .catch(() => setData({ total: 0, recent: [], stats: {} }))
      .finally(() => setLoading(false))
  }, [home, away])

  return (
    <div className="wcard">
      <div className="wc-head">
        <span className="wc-title">Head to Head</span>
        {data && <span className="wc-meta">{data.total} partidos hist\u00F3ricos</span>}
      </div>
      <div className="wc-body">
        {loading && <div style={{ color: 'var(--color-text-subtle)', textAlign: 'center', padding: 12 }}>Cargando...</div>}
        {!loading && data && data.total === 0 && (
          <div className="h2h-empty">Sin enfrentamientos en el historial disponible</div>
        )}
        {!loading && data && data.total > 0 && (
          <H2HContent data={data} home={home} away={away} />
        )}
      </div>
    </div>
  )
}

function H2HContent({ data, home, away }) {
  const { stats, recent } = data
  const total = (stats.team_a_wins + stats.draws + stats.team_b_wins) || 1
  const pA = (stats.team_a_wins / total * 100).toFixed(0)
  const pD = (stats.draws / total * 100).toFixed(0)
  const pB = (stats.team_b_wins / total * 100).toFixed(0)

  return (
    <>
      <div className="h2h-summary">
        <div className="h2h-sum-team">
          <div className="h2h-sum-name">{home.split(' ')[0]}</div>
          <div className="h2h-sum-wins">{stats.team_a_wins}</div>
          <div className="h2h-sum-lbl">victorias</div>
        </div>
        <div className="h2h-sum-center">
          <div className="h2h-sum-draws">{stats.draws}</div>
          <div className="h2h-sum-lbl">empates</div>
        </div>
        <div className="h2h-sum-team">
          <div className="h2h-sum-name">{away.split(' ')[0]}</div>
          <div className="h2h-sum-wins" style={{ color: 'var(--color-text-info)' }}>{stats.team_b_wins}</div>
          <div className="h2h-sum-lbl">victorias</div>
        </div>
      </div>
      <div className="h2h-bar-wrap">
        <div className="h2h-bar" style={{ gridTemplateColumns: `${pA}fr ${pD}fr ${pB}fr` }}>
          <div className="h2h-bar-a" /><div className="h2h-bar-d" /><div className="h2h-bar-b" />
        </div>
        <div className="h2h-goals">
          <span>{stats.team_a_goals} goles</span>
          <span>{stats.team_b_goals} goles</span>
        </div>
      </div>
      <div className="h2h-meetings">
        {[...recent].reverse().map((m, i) => {
          const isWinA = (m.home === home && m.result === 'H') || (m.home === away && m.result === 'A')
          const cls = m.result === 'D' ? 'draw' : isWinA ? 'win-a' : 'win-b'
          return (
            <div key={i} className="h2h-row">
              <span className="h2h-season">{m.season}</span>
              <span className="h2h-home">{m.home}</span>
              <span className={`h2h-score ${cls}`}>{m.fthg} - {m.ftag}</span>
              <span className="h2h-away">{m.away}</span>
            </div>
          )
        })}
      </div>
    </>
  )
}
