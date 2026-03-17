import { useState, useCallback } from 'react'
import { api } from '../services/api'

export const usePrediction = () => {
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const predict = useCallback(async (homeTeam, awayTeam) => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.predict(homeTeam, awayTeam)
      setPrediction(data)
      return data
    } catch (err) {
      const errorMessage = err.message || 'Error en la predicción'
      setError(errorMessage)
      setPrediction(null)
    } finally {
      setLoading(false)
    }
  }, [])

  const reset = useCallback(() => {
    setPrediction(null)
    setError(null)
    setLoading(false)
  }, [])

  return { prediction, loading, error, predict, reset }
}

export default usePrediction
