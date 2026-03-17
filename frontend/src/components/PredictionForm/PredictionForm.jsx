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

  useEffect(() => {
    api.teams()
      .then((data) => setTeams(data))
      .catch(() => setTeams(FALLBACK_TEAMS))
  }, [])

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

    onSubmit(homeTeam, awayTeam)
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

      <button
        type="submit"
        className="btn-submit"
        disabled={loading || !homeTeam || !awayTeam}
      >
        {loading ? 'Analyzing...' : 'Get Prediction'}
      </button>
    </form>
  )
}

export default PredictionForm
