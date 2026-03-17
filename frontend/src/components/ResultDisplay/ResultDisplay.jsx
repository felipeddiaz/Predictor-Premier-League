import './ResultDisplay.css'

export const ResultDisplay = ({ prediction }) => {
  if (!prediction) return null

  const formatPct = (value) => (value * 100).toFixed(1)

  // Detect if this is a detailed response (with odds/value betting data)
  const hasOdds = prediction.market_prob_local !== undefined

  // Normalize field names between simple and detail responses
  const homeTeam = prediction.home_team || prediction.local
  const awayTeam = prediction.away_team || prediction.away
  const outcome = prediction.predicted_outcome || prediction.prediction
  const probHome = prediction.home_win_probability ?? prediction.prob_local
  const probDraw = prediction.draw_probability ?? prediction.prob_draw
  const probAway = prediction.away_win_probability ?? prediction.prob_away
  const homeForm = prediction.home_form || prediction.form_local
  const awayForm = prediction.away_form || prediction.form_away

  // Value betting helpers
  const getEdgeClass = (modelProb, marketProb) => {
    const diff = modelProb - marketProb
    if (diff > 0.05) return 'value-positive'
    if (diff < -0.05) return 'value-negative'
    return ''
  }

  const formatEdge = (modelProb, marketProb) => {
    const diff = ((modelProb - marketProb) * 100).toFixed(1)
    return diff > 0 ? `+${diff}%` : `${diff}%`
  }

  return (
    <div className="result-container">
      <div className="match-header">
        <h2 className="team local">{homeTeam}</h2>
        <span className="vs">vs</span>
        <h2 className="team away">{awayTeam}</h2>
      </div>

      <div className="prediction-card">
        <div className="prediction-result">
          <h3>Model Prediction</h3>
          <p className="prediction-value">{outcome}</p>
          <p className="confidence">
            Confidence: <strong>{formatPct(prediction.confidence)}%</strong>
          </p>
        </div>
      </div>

      <div className="probabilities-section">
        <h3>Probabilities</h3>
        <div className="probability-bars">
          <div className="probability-bar">
            <div className="bar-header">
              <span>Home Win</span>
              <span>{formatPct(probHome)}%</span>
            </div>
            <div className="bar">
              <div
                className="fill local"
                style={{ width: `${probHome * 100}%` }}
              />
            </div>
          </div>

          <div className="probability-bar">
            <div className="bar-header">
              <span>Draw</span>
              <span>{formatPct(probDraw)}%</span>
            </div>
            <div className="bar">
              <div
                className="fill draw"
                style={{ width: `${probDraw * 100}%` }}
              />
            </div>
          </div>

          <div className="probability-bar">
            <div className="bar-header">
              <span>Away Win</span>
              <span>{formatPct(probAway)}%</span>
            </div>
            <div className="bar">
              <div
                className="fill away"
                style={{ width: `${probAway * 100}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {hasOdds && (
        <div className="value-section">
          <h3>Value Betting Analysis</h3>
          <div className="value-grid">
            <div className={`value-item ${getEdgeClass(probHome, prediction.market_prob_local)}`}>
              <span className="label">Home (1)</span>
              <div className="value-comparison">
                <span>Model: {formatPct(probHome)}%</span>
                <span>Market: {formatPct(prediction.market_prob_local)}%</span>
              </div>
              <p className="edge-value">{formatEdge(probHome, prediction.market_prob_local)}</p>
            </div>
            <div className={`value-item ${getEdgeClass(probDraw, prediction.market_prob_draw)}`}>
              <span className="label">Draw (X)</span>
              <div className="value-comparison">
                <span>Model: {formatPct(probDraw)}%</span>
                <span>Market: {formatPct(prediction.market_prob_draw)}%</span>
              </div>
              <p className="edge-value">{formatEdge(probDraw, prediction.market_prob_draw)}</p>
            </div>
            <div className={`value-item ${getEdgeClass(probAway, prediction.market_prob_away)}`}>
              <span className="label">Away (2)</span>
              <div className="value-comparison">
                <span>Model: {formatPct(probAway)}%</span>
                <span>Market: {formatPct(prediction.market_prob_away)}%</span>
              </div>
              <p className="edge-value">{formatEdge(probAway, prediction.market_prob_away)}</p>
            </div>
          </div>
        </div>
      )}

      {hasOdds && prediction.over25_prob != null && (
        <div className="binary-markets">
          <h3>Binary Markets</h3>
          <div className="binary-grid">
            <div className="binary-item">
              <span>Over 2.5 Goals</span>
              <strong>{formatPct(prediction.over25_prob)}%</strong>
            </div>
            {prediction.over35_cards_prob != null && (
              <div className="binary-item">
                <span>Over 3.5 Cards</span>
                <strong>{formatPct(prediction.over35_cards_prob)}%</strong>
              </div>
            )}
            {prediction.over95_corners_prob != null && (
              <div className="binary-item">
                <span>Over 9.5 Corners</span>
                <strong>{formatPct(prediction.over95_corners_prob)}%</strong>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="analysis-section">
        <h3>Team Form (Last 5 Matches)</h3>
        <div className="analysis-grid">
          <div className="analysis-item">
            <span className="label">{homeTeam}</span>
            <p>{homeForm || 'N/A'}</p>
          </div>

          <div className="analysis-item">
            <span className="label">{awayTeam}</span>
            <p>{awayForm || 'N/A'}</p>
          </div>
        </div>
      </div>

      <div className="disclaimer">
        <p>
          These predictions are for informational and educational purposes only.
          They do not constitute financial advice.
        </p>
      </div>
    </div>
  )
}

export default ResultDisplay
