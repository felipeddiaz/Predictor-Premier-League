import { useEffect, useState } from 'react'
import { api } from '../../services/api'
import './Header.css'

export const Header = () => {
  const [apiStatus, setApiStatus] = useState('checking')

  useEffect(() => {
    const checkAPI = async () => {
      try {
        const response = await api.health()
        setApiStatus(response.data.status)
      } catch (err) {
        setApiStatus('offline')
      }
    }

    checkAPI()
    const interval = setInterval(checkAPI, 30000)
    return () => clearInterval(interval)
  }, [])

  return (
    <header className="header">
      <div className="header-container">
        <div className="header-brand">
          <h1>⚽ Premier League Predictor</h1>
          <p className="subtitle">Predicciones IA para la Premier League</p>
        </div>

        <div className={`status-badge ${apiStatus}`}>
          {apiStatus === 'healthy' ? (
            <>
              <span className="status-dot">●</span>
              <span>API Conectada</span>
            </>
          ) : (
            <>
              <span className="status-dot">●</span>
              <span>API Offline</span>
            </>
          )}
        </div>
      </div>
    </header>
  )
}

export default Header
