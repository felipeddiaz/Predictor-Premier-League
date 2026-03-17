import { useState, useCallback } from 'react'

/**
 * Hook reutilizable para llamadas a API
 * @param {Function} apiFunction - Función de API a ejecutar
 * @param {*} initialData - Datos iniciales
 * @returns {Object} { data, loading, error, execute }
 */
export const useAPI = (apiFunction, initialData = null) => {
  const [data, setData] = useState(initialData)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const execute = useCallback(
    async (...args) => {
      setLoading(true)
      setError(null)
      try {
        const response = await apiFunction(...args)
        setData(response.data)
        return response.data
      } catch (err) {
        const errorMessage =
          err.response?.data?.detail ||
          err.message ||
          'Error desconocido'
        setError(errorMessage)
        throw err
      } finally {
        setLoading(false)
      }
    },
    [apiFunction]
  )

  return { data, loading, error, execute }
}

export default useAPI
