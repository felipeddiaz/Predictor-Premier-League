import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [predictions, setPredictions] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchPredictions()
  }, [])

  const fetchPredictions = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch('/api/predictions')
      const data = await response.json()
      setPredictions(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <header className="header">
        <h1>🔮 PL Predictor</h1>
        <p>Predicciones de la Premier League</p>
      </header>

      <main className="main">
        {loading && <p>Cargando predicciones...</p>}
        {error && <p className="error">Error: {error}</p>}

        {predictions.length > 0 ? (
          <div className="predictions-grid">
            {predictions.map((pred) => (
              <div key={pred.id} className="prediction-card">
                <h3>{pred.match}</h3>
                <p className="prediction">{pred.prediction}</p>
                <p className="confidence">Confianza: {(pred.confidence * 100).toFixed(1)}%</p>
              </div>
            ))}
          </div>
        ) : (
          <p>No hay predicciones disponibles</p>
        )}
      </main>
    </div>
  )
}

export default App
