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

  const handlePredictionSubmit = async (matchData) => {
    setDismissedError(false)
    try {
      await predict(matchData, true)
    } catch (err) {
      console.error('Prediction error:', err)
    }
  }

  const handleReset = () => {
    reset()
    setDismissedError(false)
  }

  const handleDismissError = () => {
    setDismissedError(true)
  }

  return (
    <div className="home">
      <Header />

      <main className="home-main">
        <div className="container">
          {!dismissedError && error && (
            <ErrorMessage message={error} onDismiss={handleDismissError} />
          )}

          {!prediction && !loading && (
            <>
              <div className="hero-section">
                <h2 className="hero-title">Predicciones inteligentes</h2>
                <p className="hero-subtitle">
                  Selecciona los equipos y obtiene predicciones basadas en Machine Learning
                </p>
              </div>

              <PredictionForm onSubmit={handlePredictionSubmit} loading={loading} />

              <section className="how-it-works">
                <div className="how-grid">
                  <div className="how-item">
                    <div className="how-number">1</div>
                    <div>
                      <h4>Selecciona equipos</h4>
                      <p>Elige local y visitante de la Premier League</p>
                    </div>
                  </div>
                  <div className="how-item">
                    <div className="how-number">2</div>
                    <div>
                      <h4>Agrega cuotas</h4>
                      <p>Opcionalmente ingresa las cuotas del mercado</p>
                    </div>
                  </div>
                  <div className="how-item">
                    <div className="how-number">3</div>
                    <div>
                      <h4>Analisis IA</h4>
                      <p>El modelo analiza 56+ indicadores y 10 temporadas</p>
                    </div>
                  </div>
                  <div className="how-item">
                    <div className="how-number">4</div>
                    <div>
                      <h4>Resultado</h4>
                      <p>Obtiene prediccion, probabilidades y edge vs mercado</p>
                    </div>
                  </div>
                </div>
              </section>
            </>
          )}

          {loading && <Loading message="Analizando partido..." />}

          {prediction && !loading && (
            <>
              <ResultDisplay prediction={prediction} />
              <div className="reset-area">
                <button className="btn-new-prediction" onClick={handleReset}>
                  Nueva prediccion
                </button>
              </div>
            </>
          )}
        </div>
      </main>

      <footer className="home-footer">
        <p>Premier League Predictor &middot; ML Powered &middot; Uso educativo</p>
      </footer>
    </div>
  )
}

export default Home
