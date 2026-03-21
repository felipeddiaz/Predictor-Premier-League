import './ResultDisplay.css'

const TEAM_COLORS = {
  'Arsenal': '#EF0107', 'Aston Villa': '#670E36', 'Bournemouth': '#DA291C',
  'Brentford': '#E30613', 'Brighton': '#0057B8', 'Chelsea': '#034694',
  'Crystal Palace': '#1B458F', 'Everton': '#003399', 'Fulham': '#CC0000',
  'Ipswich': '#0044AA', 'Leicester': '#003090', 'Liverpool': '#C8102E',
  'Manchester City': '#6CABDD', 'Manchester United': '#DA291C', 'Newcastle': '#241F20',
  'Nottingham': '#DD0000', 'Southampton': '#D71920', 'Tottenham': '#132257',
  'West Ham': '#7A263A', 'Wolves': '#FDB913',
}

const TEAM_SHORT = {
  'Arsenal': 'ARS', 'Aston Villa': 'AVL', 'Bournemouth': 'BOU',
  'Brentford': 'BRE', 'Brighton': 'BHA', 'Chelsea': 'CHE',
  'Crystal Palace': 'CRY', 'Everton': 'EVE', 'Fulham': 'FUL',
  'Ipswich': 'IPS', 'Leicester': 'LEI', 'Liverpool': 'LIV',
  'Manchester City': 'MCI', 'Manchester United': 'MUN', 'Newcastle': 'NEW',
  'Nottingham': 'NFO', 'Southampton': 'SOU', 'Tottenham': 'TOT',
  'West Ham': 'WHU', 'Wolves': 'WOL',
}

export const ResultDisplay = ({ prediction }) => {
  if (!prediction) return null

  const pct = (v) => (v * 100).toFixed(1)
  const homeColor = TEAM_COLORS[prediction.local] || '#3b82f6'
  const awayColor = TEAM_COLORS[prediction.away] || '#ef4444'

  const predictionLabel = {
    'Local': `Victoria ${prediction.local}`,
    'Visitante': `Victoria ${prediction.away}`,
    'Empate': 'Empate',
  }[prediction.prediction] || prediction.prediction

  const edgePositive = prediction.edge > 0
  const edgePct = (prediction.edge * 100).toFixed(1)

  return (
    <div className="results">
      {/* Match Header */}
      <div className="result-match-header">
        <div className="result-team">
          <div className="result-badge" style={{ borderColor: `${homeColor}88`, background: `${homeColor}1a` }}>
            <span>{TEAM_SHORT[prediction.local] || '?'}</span>
          </div>
          <span className="result-team-name">{prediction.local}</span>
          <span className="result-team-tag">LOCAL</span>
        </div>

        <div className="result-vs-area">
          <span className="result-vs">VS</span>
        </div>

        <div className="result-team">
          <div className="result-badge" style={{ borderColor: `${awayColor}88`, background: `${awayColor}1a` }}>
            <span>{TEAM_SHORT[prediction.away] || '?'}</span>
          </div>
          <span className="result-team-name">{prediction.away}</span>
          <span className="result-team-tag">VISITANTE</span>
        </div>
      </div>

      {/* Prediction Result */}
      <div className="prediction-hero">
        <span className="prediction-hero-label">Prediccion del modelo</span>
        <span className="prediction-hero-value">{predictionLabel}</span>
        <div className="prediction-hero-confidence">
          <div className="confidence-ring">
            <svg viewBox="0 0 36 36" className="confidence-svg">
              <path
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                fill="none"
                stroke="var(--border)"
                strokeWidth="3"
              />
              <path
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                fill="none"
                stroke="var(--primary)"
                strokeWidth="3"
                strokeDasharray={`${prediction.confidence * 100}, 100`}
                strokeLinecap="round"
              />
            </svg>
            <span className="confidence-number">{pct(prediction.confidence)}%</span>
          </div>
          <span className="confidence-label">Confianza</span>
        </div>
      </div>

      {/* Probability Comparison */}
      <div className="prob-comparison">
        <div className="prob-section">
          <span className="prob-section-title">Modelo</span>
          <div className="prob-bars-group">
            <ProbBar label="1" value={prediction.prob_local} color={homeColor} />
            <ProbBar label="X" value={prediction.prob_draw} color="#f59e0b" />
            <ProbBar label="2" value={prediction.prob_away} color={awayColor} />
          </div>
        </div>

        {prediction.market_prob_local != null && (
          <div className="prob-section">
            <span className="prob-section-title">Mercado</span>
            <div className="prob-bars-group">
              <ProbBar label="1" value={prediction.market_prob_local} color={homeColor} />
              <ProbBar label="X" value={prediction.market_prob_draw} color="#f59e0b" />
              <ProbBar label="2" value={prediction.market_prob_away} color={awayColor} />
            </div>
          </div>
        )}
      </div>

      {/* Edge + Form Row */}
      <div className="stats-row">
        <div className={`stat-chip ${edgePositive ? 'edge-positive' : 'edge-negative'}`}>
          <span className="stat-chip-label">Edge</span>
          <span className="stat-chip-value">
            {edgePositive ? '+' : ''}{edgePct}%
          </span>
        </div>

        <div className="stat-chip">
          <span className="stat-chip-label">Forma {prediction.local}</span>
          <span className="stat-chip-value">{prediction.form_local || 'N/A'}</span>
        </div>

        <div className="stat-chip">
          <span className="stat-chip-label">Forma {prediction.away}</span>
          <span className="stat-chip-value">{prediction.form_away || 'N/A'}</span>
        </div>
      </div>

      {/* Binary Markets */}
      {(prediction.over25_prob || prediction.over35_cards_prob || prediction.over95_corners_prob) && (
        <div className="binary-section">
          <span className="binary-title">Mercados adicionales</span>
          <div className="binary-chips">
            {prediction.over25_prob != null && (
              <BinaryChip label="Over 2.5 Goles" value={prediction.over25_prob} />
            )}
            {prediction.over35_cards_prob != null && (
              <BinaryChip label="Over 3.5 Tarjetas" value={prediction.over35_cards_prob} />
            )}
            {prediction.over95_corners_prob != null && (
              <BinaryChip label="Over 9.5 Corners" value={prediction.over95_corners_prob} />
            )}
          </div>
        </div>
      )}

      {/* Disclaimer */}
      <p className="result-disclaimer">
        Predicciones informativas y educativas. No constituyen consejo financiero.
      </p>
    </div>
  )
}

const ProbBar = ({ label, value, color }) => (
  <div className="prob-bar-row">
    <span className="prob-label">{label}</span>
    <div className="prob-track">
      <div
        className="prob-fill"
        style={{ width: `${value * 100}%`, background: color }}
      />
    </div>
    <span className="prob-pct">{(value * 100).toFixed(1)}%</span>
  </div>
)

const BinaryChip = ({ label, value }) => {
  const pct = (value * 100).toFixed(0)
  const isHigh = value >= 0.55
  return (
    <div className={`binary-chip ${isHigh ? 'high' : 'low'}`}>
      <span className="binary-chip-label">{label}</span>
      <span className="binary-chip-value">{pct}%</span>
    </div>
  )
}

export default ResultDisplay
