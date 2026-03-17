import './ResultDisplay.css'

export const ResultDisplay = ({ prediction }) => {
  if (!prediction) return null

  const formatPercentage = (value) => (value * 100).toFixed(1)

  return (
    <div className="result-container">
      <div className="match-header">
        <h2 className="team local">{prediction.local}</h2>
        <span className="vs">vs</span>
        <h2 className="team away">{prediction.away}</h2>
      </div>

      <div className="prediction-card">
        <div className="prediction-result">
          <h3>Predicción del Modelo</h3>
          <p className="prediction-value">{prediction.prediction}</p>
          <p className="confidence">
            Confianza: <strong>{formatPercentage(prediction.confidence)}%</strong>
          </p>
        </div>
      </div>

      <div className="probabilities-section">
        <h3>Probabilidades del Modelo</h3>
        <div className="probability-bars">
          <div className="probability-bar">
            <div className="bar-header">
              <span>Local</span>
              <span>{formatPercentage(prediction.prob_local)}%</span>
            </div>
            <div className="bar">
              <div
                className="fill local"
                style={{ width: `${prediction.prob_local * 100}%` }}
              />
            </div>
          </div>

          <div className="probability-bar">
            <div className="bar-header">
              <span>Empate</span>
              <span>{formatPercentage(prediction.prob_draw)}%</span>
            </div>
            <div className="bar">
              <div
                className="fill draw"
                style={{ width: `${prediction.prob_draw * 100}%` }}
              />
            </div>
          </div>

          <div className="probability-bar">
            <div className="bar-header">
              <span>Visitante</span>
              <span>{formatPercentage(prediction.prob_away)}%</span>
            </div>
            <div className="bar">
              <div
                className="fill away"
                style={{ width: `${prediction.prob_away * 100}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {prediction.market_prob_local && (
        <div className="probabilities-section">
          <h3>Probabilidades del Mercado</h3>
          <div className="probability-bars">
            <div className="probability-bar">
              <div className="bar-header">
                <span>Local</span>
                <span>{formatPercentage(prediction.market_prob_local)}%</span>
              </div>
              <div className="bar">
                <div
                  className="fill local"
                  style={{ width: `${prediction.market_prob_local * 100}%` }}
                />
              </div>
            </div>

            <div className="probability-bar">
              <div className="bar-header">
                <span>Empate</span>
                <span>{formatPercentage(prediction.market_prob_draw)}%</span>
              </div>
              <div className="bar">
                <div
                  className="fill draw"
                  style={{ width: `${prediction.market_prob_draw * 100}%` }}
                />
              </div>
            </div>

            <div className="probability-bar">
              <div className="bar-header">
                <span>Visitante</span>
                <span>{formatPercentage(prediction.market_prob_away)}%</span>
              </div>
              <div className="bar">
                <div
                  className="fill away"
                  style={{ width: `${prediction.market_prob_away * 100}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="analysis-section">
        <h3>Análisis</h3>
        <div className="analysis-grid">
          <div className="analysis-item">
            <span className="label">Forma Local</span>
            <p>{prediction.form_local || 'N/A'}</p>
          </div>

          <div className="analysis-item">
            <span className="label">Forma Visitante</span>
            <p>{prediction.form_away || 'N/A'}</p>
          </div>

          <div className="analysis-item">
            <span className="label">Edge (Valor)</span>
            <p className={prediction.edge > 0 ? 'positive' : 'negative'}>
              {prediction.edge > 0 ? '+' : ''}{(prediction.edge * 100).toFixed(2)}%
            </p>
          </div>
        </div>
      </div>

      {(prediction.over25_prob ||
        prediction.over35_cards_prob ||
        prediction.over95_corners_prob) && (
        <div className="binary-markets">
          <h3>Mercados Binarios</h3>
          <div className="binary-grid">
            {prediction.over25_prob && (
              <div className="binary-item">
                <span>Over 2.5 Goles</span>
                <strong>{formatPercentage(prediction.over25_prob)}%</strong>
              </div>
            )}
            {prediction.over35_cards_prob && (
              <div className="binary-item">
                <span>Over 3.5 Tarjetas</span>
                <strong>{formatPercentage(prediction.over35_cards_prob)}%</strong>
              </div>
            )}
            {prediction.over95_corners_prob && (
              <div className="binary-item">
                <span>Over 9.5 Corners</span>
                <strong>{formatPercentage(prediction.over95_corners_prob)}%</strong>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="disclaimer">
        <p>
          💡 <strong>Disclaimer:</strong> Estas predicciones son informativas y educativas.
          No constituyen consejo financiero.
        </p>
      </div>
    </div>
  )
}

export default ResultDisplay
