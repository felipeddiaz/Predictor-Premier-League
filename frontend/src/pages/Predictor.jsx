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
              <span style={{ fontSize: 10, color: 'var(--ink4)' }}>{matches.length} partidos</span>
            </div>
            {meta?.last_updated && (
              <div style={{ fontSize: 10, color: 'var(--ink4)', marginBottom: 6 }}>
                Actualizado: {new Date(meta.last_updated).toLocaleString()}
              </div>
            )}
            <button className="btn-add" onClick={() => loadMatches(true)}
              style={{ borderStyle: 'solid', background: 'var(--bg)' }}>
              Actualizar cuotas
            </button>
          </div>
          <div className="sb-list">
            {!matches.length && !loading && (
              <div style={{ padding: 16, textAlign: 'center', color: 'var(--ink4)', fontSize: 12 }}>
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

/* ── Sidebar Match Item ── */
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

/* ── Match Detail Panel ── */
function MatchDetail({ match: m }) {
  const pred = m.pred
  const odds = m.odds || {}
  const bh = odds.home || 2.80
  const bd = odds.draw || 3.40
  const ba = odds.away || 2.80
  const [statsTeam, setStatsTeam] = useState(null) // { name, isLocal }

  return (
    <>
      {/* HEADER CARD */}
      <div className="w-header">
        <div className="wh-team" style={{ cursor: 'pointer' }} onClick={() => setStatsTeam({ name: m.home, isLocal: true })}>
          <TeamBadge name={m.home} size={48} style={{ borderRadius: 12 }} />
          <div className="wh-name" style={{ textDecoration: 'underline dotted', textUnderlineOffset: 3 }}>{m.home}</div>
          <div style={{ fontSize: 10, color: 'var(--ink4)' }}>Local</div>
        </div>

        <div className="wh-center">
          <div className="wh-league">Premier League</div>
          <div className="wh-vs" style={{ fontSize: 16, fontWeight: 800, margin: '4px 0' }}>vs</div>
          <div className={`wh-status${pred ? ' done' : ''}`}>
            {'\u25CF'} {pred ? pred.prediction === 'H' ? 'Local' : pred.prediction === 'D' ? 'Empate' : 'Visitante' : 'Sin predicci\u00F3n'}
          </div>
          {m.commence_time && (
            <div style={{ fontSize: 10, color: 'var(--ink4)', marginTop: 4 }}>
              {new Date(m.commence_time).toLocaleDateString()} {new Date(m.commence_time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </div>
          )}
          <div className="wh-odds" style={{ marginTop: 8 }}>
            {[['1', bh], ['X', bd], ['2', ba]].map(([lbl, val]) => (
              <div className="wh-odd-box" key={lbl}>
                <span className="wh-odd-lbl">{lbl}</span>
                <div className="wh-odd-inp" style={{ cursor: 'default' }}>{val}</div>
              </div>
            ))}
          </div>
          {m.odds_changed && m.odds_previous && (
            <div style={{ fontSize: 9, color: 'var(--amb)', fontWeight: 700, marginTop: 4 }}>
              Cuotas actualizadas (antes: {m.odds_previous.home} / {m.odds_previous.draw} / {m.odds_previous.away})
            </div>
          )}
          {odds.n_bookmakers > 0 && (
            <div style={{ fontSize: 9, color: 'var(--ink4)', marginTop: 2 }}>
              Promedio de {odds.n_bookmakers} casas de apuestas
            </div>
          )}
        </div>

        <div className="wh-team" style={{ cursor: 'pointer' }} onClick={() => setStatsTeam({ name: m.away, isLocal: false })}>
          <TeamBadge name={m.away} size={48} style={{ borderRadius: 12, opacity: .85 }} />
          <div className="wh-name" style={{ textDecoration: 'underline dotted', textUnderlineOffset: 3 }}>{m.away}</div>
          <div style={{ fontSize: 10, color: 'var(--ink4)' }}>Visitante</div>
        </div>
      </div>

      {/* ANALYSIS WIDGETS */}
      {pred && <AnalysisWidgets pred={pred} match={m} bh={bh} bd={bd} ba={ba} />}

      {/* H2H WIDGET */}
      <H2HWidget home={m.home} away={m.away} />

      {!pred && (
        <div className="wcard">
          <div className="wc-body" style={{ textAlign: 'center', padding: 20, color: 'var(--ink4)' }}>
            No se pudo generar predicci\u00F3n para este partido
          </div>
        </div>
      )}

      {/* TEAM STATS MODAL */}
      {statsTeam && (
        <TeamStatsModal
          name={statsTeam.name}
          isLocal={statsTeam.isLocal}
          onClose={() => setStatsTeam(null)}
        />
      )}
    </>
  )
}

/* ── Team Stats Modal ── */
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

  const fmt = (v, dec = 2) => v != null ? (+v).toFixed(dec) : '—'

  const ATTACK = [
    ['Goles prom.', 'AvgGoals', 1],
    ['xG prom.', 'xG_Avg', 2],
    ['xG global', 'xG_Global', 2],
    ['Disparos a puerta', 'AvgShotsTarget', 1],
    ['Goles (ult.5)', 'GoalsFor5', 1],
    ['SoR (ult.5)', 'SoR5', 2],
  ]
  const DEFENSE = [
    ['xGA prom.', 'xGA_Avg', 2],
    ['xGA global', 'xGA_Global', 2],
    ['Goles en contra (ult.5)', 'GoalsAgainst5', 1],
  ]
  const FORM = [
    ['Posici\u00F3n', 'Position', 0],
    ['Puntos', 'Points', 0],
    ['Racha', 'Streak', 0],
    ['Win Rate (ult.5)', 'WinRate5', 2],
    ['Pts (ult.5)', 'Pts5', 1],
    ['Momentum', 'Form_Momentum', 2],
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
          {loading && <div style={{ color: 'var(--ink4)', textAlign: 'center', padding: 20 }}>Cargando...</div>}
          {!loading && stats && (
            <>
              {/* Forma */}
              <div className="ts-group">
                <div className="ts-group-lbl">Forma reciente</div>
                <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
                  {[['W', stats.Form_W, 'var(--grn)'], ['D', stats.Form_D, 'var(--amb)'], ['L', stats.Form_L, 'var(--red)']].map(([l, v, c]) => (
                    <div key={l} style={{ flex: 1, background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 8, padding: '8px 0', textAlign: 'center' }}>
                      <div style={{ fontSize: 18, fontWeight: 800, fontFamily: 'var(--mono)', color: c }}>{fmt(v, 1)}</div>
                      <div style={{ fontSize: 9, color: 'var(--ink4)', fontWeight: 600, marginTop: 2 }}>{l === 'W' ? 'Victorias' : l === 'D' ? 'Empates' : 'Derrotas'}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Ataque */}
              <div className="ts-group">
                <div className="ts-group-lbl">Ataque</div>
                <div className="ts-stats-grid">
                  {ATTACK.map(([lbl, key, dec]) => stats[key] != null && (
                    <div key={key} className="ts-stat">
                      <div className="ts-stat-lbl">{lbl}</div>
                      <div className="ts-stat-val" style={{ color: 'var(--grn)', fontSize: 16 }}>{fmt(stats[key], dec)}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Defensa */}
              <div className="ts-group">
                <div className="ts-group-lbl">Defensa</div>
                <div className="ts-stats-grid">
                  {DEFENSE.map(([lbl, key, dec]) => stats[key] != null && (
                    <div key={key} className="ts-stat">
                      <div className="ts-stat-lbl">{lbl}</div>
                      <div className="ts-stat-val" style={{ color: 'var(--red)', fontSize: 16 }}>{fmt(stats[key], dec)}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Tabla y forma */}
              <div className="ts-group">
                <div className="ts-group-lbl">Tabla y tendencia</div>
                <div className="ts-stats-grid">
                  {FORM.map(([lbl, key, dec]) => stats[key] != null && (
                    <div key={key} className="ts-stat">
                      <div className="ts-stat-lbl">{lbl}</div>
                      <div className="ts-stat-val" style={{ fontSize: 16 }}>{fmt(stats[key], dec)}</div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

/* ── H2H Widget ── */
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
        {loading && <div style={{ color: 'var(--ink4)', textAlign: 'center', padding: 12 }}>Cargando...</div>}
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
  const total = stats.team_a_wins + stats.draws + stats.team_b_wins || 1
  const pA = (stats.team_a_wins / total * 100).toFixed(0)
  const pD = (stats.draws / total * 100).toFixed(0)
  const pB = (stats.team_b_wins / total * 100).toFixed(0)

  return (
    <>
      {/* Summary bar */}
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
          <div className="h2h-sum-wins" style={{ color: 'var(--blue)' }}>{stats.team_b_wins}</div>
          <div className="h2h-sum-lbl">victorias</div>
        </div>
      </div>

      {/* % bar */}
      <div className="h2h-bar-wrap">
        <div className="h2h-bar" style={{ gridTemplateColumns: `${pA}fr ${pD}fr ${pB}fr` }}>
          <div className="h2h-bar-a" />
          <div className="h2h-bar-d" />
          <div className="h2h-bar-b" />
        </div>
        <div className="h2h-goals">
          <span>{stats.team_a_goals} goles</span>
          <span>{stats.team_b_goals} goles</span>
        </div>
      </div>

      {/* Recent meetings */}
      <div className="h2h-meetings">
        {[...recent].reverse().map((m, i) => {
          const isHomeWin = (m.home === home && m.result === 'H') || (m.home === away && m.result === 'A')
          const isAwayWin = (m.home === away && m.result === 'H') || (m.home === home && m.result === 'A')
          const cls = m.result === 'D' ? 'draw' : isHomeWin ? 'win-a' : 'win-b'
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

/* ── Analysis Widgets ── */
function AnalysisWidgets({ pred, match: m, bh, bd, ba }) {
  const mkt = pred.mktH != null
    ? { h: pred.mktH, d: pred.mktD, a: pred.mktA }
    : (() => { const ip = impliedProbs(bh, bd, ba); return { h: ip.home, d: ip.draw, a: ip.away } })()

  const rawH = 1/bh, rawD = 1/bd, rawA = 1/ba
  const ov = rawH + rawD + rawA
  const vig = +((ov - 1) * 100).toFixed(1)

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

  const conf = entropyConfidence(pred.probH, pred.probD, pred.probA)
  const maxEdge = Math.max(...[eH, eD, eA].map(e => Math.abs(e)), 10)

  const ranked = [
    { k: 'H', name: 'Local', ev: evH, edge: eH, stake: kH, rec: rH, odds: bh, model: pred.probH, mkt: mkt.h },
    { k: 'D', name: 'Empate', ev: evD, edge: eD, stake: kD, rec: rD, odds: bd, model: pred.probD, mkt: mkt.d },
    { k: 'A', name: 'Visitante', ev: evA, edge: eA, stake: kA, rec: rA, odds: ba, model: pred.probA, mkt: mkt.a },
  ].sort((a, b) => b.ev - a.ev)

  const probData = [
    { k: 'H', label: 'Local', prob: pred.probH, mktP: mkt.h, edge: eH },
    { k: 'D', label: 'Empate', prob: pred.probD, mktP: mkt.d, edge: eD },
    { k: 'A', label: 'Visitante', prob: pred.probA, mktP: mkt.a, edge: eA },
  ]
  const bestProb = Math.max(pred.probH, pred.probD, pred.probA)

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
              {'\u2713'} XGBoost
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
              gridTemplateColumns: `${pred.probH}fr ${pred.probD}fr ${pred.probA}fr`,
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
            <span className="wc-meta">{'\u00DA'}ltimos 5</span>
          </div>
          <div className="wc-body">
            <div className="forma-section">
              {[{ name: m.home, form: pred.formHome }, { name: m.away, form: pred.formAway }].map(({ name, form }, i) => (
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

            {(pred.over25 != null || pred.over35cards != null || pred.over95corners != null) && (
              <div style={{ marginTop: 10, borderTop: '1px solid var(--border)', paddingTop: 8 }}>
                <div style={{ fontSize: 9, fontWeight: 700, color: 'var(--ink4)', textTransform: 'uppercase', letterSpacing: '.5px', marginBottom: 6 }}>
                  Mercados binarios
                </div>
                <div className="stats-grid">
                  {pred.over25 != null && (
                    <div className="stat-block">
                      <div className="sb-lbl">Over 2.5</div>
                      <div className="sb-home">{pred.over25}%</div>
                    </div>
                  )}
                  {pred.over35cards != null && (
                    <div className="stat-block">
                      <div className="sb-lbl">+3.5 Tarj.</div>
                      <div className="sb-home">{pred.over35cards}%</div>
                    </div>
                  )}
                  {pred.over95corners != null && (
                    <div className="stat-block">
                      <div className="sb-lbl">+9.5 Corn.</div>
                      <div className="sb-home">{pred.over95corners}%</div>
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
