import { useState, useEffect } from 'react'
import { api } from '../../services/api'
import './PredictionForm.css'

const FALLBACK_TEAMS = [
  'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
  'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Ipswich',
  'Leicester', 'Liverpool', 'Man City', 'Man United', 'Newcastle',
  "Nott'm Forest", 'Southampton', 'Tottenham', 'West Ham', 'Wolves',
]

export const PredictionForm = ({ onSubmit, loading }) => {
  const [homeTeam, setHomeTeam] = useState('')
  const [awayTeam, setAwayTeam] = useState('')
  const [teams, setTeams] = useState(FALLBACK_TEAMS)
  const [showOdds, setShowOdds] = useState(false)
  const [cuotaH, setCuotaH] = useState('')
  const [cuotaD, setCuotaD] = useState('')
  const [cuotaA, setCuotaA] = useState('')

  useEffect(() => {
    api.teams()
      .then((data) => setTeams(data))
      .catch(() => setTeams(FALLBACK_TEAMS))
  }, [])

  const hasValidOdds = () => {
    if (!cuotaH || !cuotaD || !cuotaA) return false
    const h = parseFloat(cuotaH)
    const d = parseFloat(cuotaD)
    const a = parseFloat(cuotaA)
    return h > 1 && d > 1 && a > 1
  }

  const handleSubmit = (e) => {
    e.preventDefault()

    if (!homeTeam || !awayTeam) {
      alert('Selecciona ambos equipos')
      return
    }

    if (homeTeam === awayTeam) {
      alert('Los equipos no pueden ser iguales')
      return
    }

    const odds = showOdds && hasValidOdds()
      ? { cuota_h: parseFloat(cuotaH), cuota_d: parseFloat(cuotaD), cuota_a: parseFloat(cuotaA) }
      : null

    onSubmit(homeTeam, awayTeam, odds)
  }

  return (
    <form className="prediction-form" onSubmit={handleSubmit}>
      <div className="form-row">
        <div className="form-group">
          <label htmlFor="home-team">Home Team</label>
          <select
            id="home-team"
            value={homeTeam}
            onChange={(e) => setHomeTeam(e.target.value)}
            className="form-input"
            required
          >
            <option value="">-- Select home team --</option>
            {teams.map((team) => (
              <option key={team} value={team}>
                {team}
              </option>
            ))}
          </select>
        </div>

        <div className="vs-separator">VS</div>

        <div className="form-group">
          <label htmlFor="away-team">Away Team</label>
          <select
            id="away-team"
            value={awayTeam}
            onChange={(e) => setAwayTeam(e.target.value)}
            className="form-input"
            required
          >
            <option value="">-- Select away team --</option>
            {teams.map((team) => (
              <option key={team} value={team}>
                {team}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="odds-toggle">
        <button
          type="button"
          className={`btn-toggle ${showOdds ? 'active' : ''}`}
          onClick={() => setShowOdds(!showOdds)}
        >
          {showOdds ? 'Hide Odds' : 'Add Odds (Value Betting)'}
        </button>
        {!showOdds && (
          <span className="odds-hint">Add betting odds to detect value bets</span>
        )}
      </div>

      {showOdds && (
        <div className="cuotas-section">
          <h3>Betting Odds (1X2)</h3>
          <div className="cuotas-grid">
            <div className="form-group">
              <label htmlFor="cuota-h">Home (1)</label>
              <input
                id="cuota-h"
                type="number"
                step="0.01"
                min="1.01"
                placeholder="e.g. 2.10"
                value={cuotaH}
                onChange={(e) => setCuotaH(e.target.value)}
                className="form-input"
              />
            </div>
            <div className="form-group">
              <label htmlFor="cuota-d">Draw (X)</label>
              <input
                id="cuota-d"
                type="number"
                step="0.01"
                min="1.01"
                placeholder="e.g. 3.40"
                value={cuotaD}
                onChange={(e) => setCuotaD(e.target.value)}
                className="form-input"
              />
            </div>
            <div className="form-group">
              <label htmlFor="cuota-a">Away (2)</label>
              <input
                id="cuota-a"
                type="number"
                step="0.01"
                min="1.01"
                placeholder="e.g. 3.20"
                value={cuotaA}
                onChange={(e) => setCuotaA(e.target.value)}
                className="form-input"
              />
            </div>
          </div>
          {showOdds && !hasValidOdds() && (cuotaH || cuotaD || cuotaA) && (
            <p className="odds-warning">All three odds must be greater than 1.00</p>
          )}
        </div>
      )}

      <button
        type="submit"
        className="btn-submit"
        disabled={loading || !homeTeam || !awayTeam}
      >
        {loading ? 'Analyzing...' : showOdds && hasValidOdds() ? 'Get Prediction + Value Analysis' : 'Get Prediction'}
      </button>
    </form>
  )
}

export default PredictionForm
