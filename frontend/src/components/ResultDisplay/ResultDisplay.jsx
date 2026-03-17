import './ResultDisplay.css'

export const ResultDisplay = ({ prediction }) => {
  if (!prediction) return null

  const formatPct = (value) => (value * 100).toFixed(1)

  return (
    <div className="result-container">
      <div className="match-header">
        <h2 className="team local">{prediction.home_team}</h2>
        <span className="vs">vs</span>
        <h2 className="team away">{prediction.away_team}</h2>
      </div>

      <div className="prediction-card">
        <div className="prediction-result">
          <h3>Model Prediction</h3>
          <p className="prediction-value">{prediction.predicted_outcome}</p>
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
              <span>{formatPct(prediction.home_win_probability)}%</span>
            </div>
            <div className="bar">
              <div
                className="fill local"
                style={{ width: `${prediction.home_win_probability * 100}%` }}
              />
            </div>
          </div>

          <div className="probability-bar">
            <div className="bar-header">
              <span>Draw</span>
              <span>{formatPct(prediction.draw_probability)}%</span>
            </div>
            <div className="bar">
              <div
                className="fill draw"
                style={{ width: `${prediction.draw_probability * 100}%` }}
              />
            </div>
          </div>

          <div className="probability-bar">
            <div className="bar-header">
              <span>Away Win</span>
              <span>{formatPct(prediction.away_win_probability)}%</span>
            </div>
            <div className="bar">
              <div
                className="fill away"
                style={{ width: `${prediction.away_win_probability * 100}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      <div className="analysis-section">
        <h3>Team Form (Last 5 Matches)</h3>
        <div className="analysis-grid">
          <div className="analysis-item">
            <span className="label">{prediction.home_team}</span>
            <p>{prediction.home_form || 'N/A'}</p>
          </div>

          <div className="analysis-item">
            <span className="label">{prediction.away_team}</span>
            <p>{prediction.away_form || 'N/A'}</p>
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
