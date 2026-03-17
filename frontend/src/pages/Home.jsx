import { useState } from 'react'
import Header from '../components/Header/Header'
import PredictionForm from '../components/PredictionForm/PredictionForm'
import ResultDisplay from '../components/ResultDisplay/ResultDisplay'
import Loading from '../components/Loading/Loading'
import ErrorMessage from '../components/ErrorMessage/ErrorMessage'
import { usePrediction } from '../hooks/usePrediction'
import './Home.css'

export const Home = () => {
  const { prediction, loading, error, predict, reset } = usePrediction()
  const [dismissedError, setDismissedError] = useState(false)

  const handlePredictionSubmit = async (homeTeam, awayTeam) => {
    setDismissedError(false)
    try {
      await predict(homeTeam, awayTeam)
    } catch (err) {
      console.error('Prediction error:', err)
    }
  }

  const handleReset = () => {
    reset()
    setDismissedError(false)
  }

  return (
    <div className="home">
      <Header />

      <main className="home-main">
        <div className="container">
          {!dismissedError && error && (
            <ErrorMessage message={error} onDismiss={() => setDismissedError(true)} />
          )}

          {!prediction && (
            <section className="form-section">
              <h2>Select teams to get a prediction</h2>
              <PredictionForm onSubmit={handlePredictionSubmit} loading={loading} />
            </section>
          )}

          {loading && <Loading message="Analyzing match..." />}

          {prediction && !loading && (
            <section className="results-section">
              <ResultDisplay prediction={prediction} />

              <div className="action-buttons">
                <button className="btn-reset" onClick={handleReset}>
                  New Prediction
                </button>
              </div>
            </section>
          )}

          {!prediction && !loading && (
            <section className="info-section">
              <h2>How does it work?</h2>
              <div className="features-grid">
                <div className="feature-card">
                  <h3>Machine Learning</h3>
                  <p>Models trained with 10+ seasons of historical Premier League data</p>
                </div>

                <div className="feature-card">
                  <h3>56+ Indicators</h3>
                  <p>Deep analysis of team stats, xG, Elo ratings, and form</p>
                </div>

                <div className="feature-card">
                  <h3>Head-to-Head</h3>
                  <p>Historical matchup data between the selected teams</p>
                </div>

                <div className="feature-card">
                  <h3>Real-Time</h3>
                  <p>Instant predictions based on the latest available data</p>
                </div>
              </div>
            </section>
          )}
        </div>
      </main>

      <footer className="home-footer">
        <p>Premier League Predictor - ML Powered - Educational Purpose Only</p>
      </footer>
    </div>
  )
}

export default Home
