import { useState, useCallback } from 'react'
import { api } from '../services/api'

/**
 * Hook especializado para manejar predicciones
 * @returns {Object} { prediction, loading, error, predict, reset }
 */
export const usePrediction = () => {
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const predict = useCallback(async (matchData, detailed = true) => {
    setLoading(true)
    setError(null)
    try {
      const endpoint = detailed ? api.predictDetail : api.predict
      const response = await endpoint(matchData)

      // El endpoint devuelve un array, pero tomamos el primer elemento
      const predictionData = Array.isArray(response.data)
        ? response.data[0]
        : response.data

      setPrediction(predictionData)
      return predictionData
    } catch (err) {
      const errorMessage =
        err.response?.data?.detail ||
        err.message ||
        'Error en la predicción'
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
