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
      <div className="header-inner">
        <div className="header-logo">
          <div className="logo-icon">PL</div>
          <div>
            <h1>Predictor</h1>
            <span className="header-tagline">Premier League AI</span>
          </div>
        </div>

        <div className={`api-status ${apiStatus}`}>
          <span className="api-dot" />
          <span className="api-label">
            {apiStatus === 'healthy' ? 'Online' : apiStatus === 'checking' ? 'Conectando...' : 'Offline'}
          </span>
        </div>
      </div>
    </header>
  )
}

export default Header
