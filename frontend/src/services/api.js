import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Crear instancia de axios
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Interceptor para manejo de errores
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error)
    throw error
  }
)

// ==================== ENDPOINTS ====================

export const api = {
  // Health check
  health: () => apiClient.get('/health'),

  // Root info
  root: () => apiClient.get('/'),

  // Status
  status: () => apiClient.get('/status'),

  // Predicción simple
  predict: (matchData) => apiClient.post('/predictions', matchData),

  // Predicción detallada
  predictDetail: (matchData) => apiClient.post('/predictions/detail', matchData),

  // Jornada
  gameweek: () => apiClient.get('/gameweek'),

  // Equipos
  teams: () => apiClient.get('/teams'),

  // Historial
  history: (team, limit = 10) =>
    apiClient.get(`/history/${team}`, { params: { limit } }),
}

export default apiClient
