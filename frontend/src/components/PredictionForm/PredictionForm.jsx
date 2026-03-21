import { useState, useMemo } from 'react'
import './PredictionForm.css'

const TEAMS = [
  'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
  'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Ipswich',
  'Leicester', 'Liverpool', 'Manchester City', 'Manchester United',
  'Newcastle', 'Nottingham', 'Southampton', 'Tottenham', 'West Ham', 'Wolves',
]

const TEAM_SHORT = {
  'Arsenal': 'ARS', 'Aston Villa': 'AVL', 'Bournemouth': 'BOU',
  'Brentford': 'BRE', 'Brighton': 'BHA', 'Chelsea': 'CHE',
  'Crystal Palace': 'CRY', 'Everton': 'EVE', 'Fulham': 'FUL',
  'Ipswich': 'IPS', 'Leicester': 'LEI', 'Liverpool': 'LIV',
  'Manchester City': 'MCI', 'Manchester United': 'MUN', 'Newcastle': 'NEW',
  'Nottingham': 'NFO', 'Southampton': 'SOU', 'Tottenham': 'TOT',
  'West Ham': 'WHU', 'Wolves': 'WOL',
}

const TEAM_COLORS = {
  'Arsenal': '#EF0107', 'Aston Villa': '#670E36', 'Bournemouth': '#DA291C',
  'Brentford': '#E30613', 'Brighton': '#0057B8', 'Chelsea': '#034694',
  'Crystal Palace': '#1B458F', 'Everton': '#003399', 'Fulham': '#CC0000',
  'Ipswich': '#0044AA', 'Leicester': '#003090', 'Liverpool': '#C8102E',
  'Manchester City': '#6CABDD', 'Manchester United': '#DA291C', 'Newcastle': '#241F20',
  'Nottingham': '#DD0000', 'Southampton': '#D71920', 'Tottenham': '#132257',
  'West Ham': '#7A263A', 'Wolves': '#FDB913',
}

export const PredictionForm = ({ onSubmit, loading }) => {
  const [formData, setFormData] = useState({
    local: '',
    visitante: '',
    cuota_h: '2.00',
    cuota_d: '3.40',
    cuota_a: '3.20',
  })
  const [showOdds, setShowOdds] = useState(false)
  const [validationError, setValidationError] = useState('')

  const impliedProbs = useMemo(() => {
    const h = parseFloat(formData.cuota_h) || 0
    const d = parseFloat(formData.cuota_d) || 0
    const a = parseFloat(formData.cuota_a) || 0
    if (h <= 1 || d <= 1 || a <= 1) return null
    const total = (1/h) + (1/d) + (1/a)
    return {
      home: ((1/h) / total * 100).toFixed(0),
      draw: ((1/d) / total * 100).toFixed(0),
      away: ((1/a) / total * 100).toFixed(0),
      margin: ((total - 1) * 100).toFixed(1),
    }
  }, [formData.cuota_h, formData.cuota_d, formData.cuota_a])

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData((prev) => ({ ...prev, [name]: value }))
    setValidationError('')
  }

  const handleSubmit = (e) => {
    e.preventDefault()

    if (!formData.local || !formData.visitante) {
      setValidationError('Selecciona ambos equipos')
      return
    }

    if (formData.local === formData.visitante) {
      setValidationError('Los equipos no pueden ser iguales')
      return
    }

    onSubmit({
      local: formData.local,
      visitante: formData.visitante,
      cuota_h: parseFloat(formData.cuota_h),
      cuota_d: parseFloat(formData.cuota_d),
      cuota_a: parseFloat(formData.cuota_a),
    })
  }

  const awayTeams = TEAMS.filter(t => t !== formData.local)
  const homeTeams = TEAMS.filter(t => t !== formData.visitante)

  return (
    <form className="match-form" onSubmit={handleSubmit}>
      {/* Match Card */}
      <div className="match-card">
        <div className="match-card-header">
          <span className="match-label">Configurar partido</span>
        </div>

        <div className="match-setup">
          {/* Home Team */}
          <div className="team-side home-side">
            <div
              className="team-badge"
              style={{
                background: formData.local
                  ? `linear-gradient(135deg, ${TEAM_COLORS[formData.local]}33, ${TEAM_COLORS[formData.local]}11)`
                  : 'var(--bg-elevated)',
                borderColor: formData.local
                  ? `${TEAM_COLORS[formData.local]}66`
                  : 'var(--border-light)',
              }}
            >
              <span className="badge-text">
                {formData.local ? TEAM_SHORT[formData.local] : '?'}
              </span>
            </div>
            <select
              name="local"
              value={formData.local}
              onChange={handleChange}
              className="team-select"
              required
            >
              <option value="">Equipo local</option>
              {homeTeams.map((team) => (
                <option key={team} value={team}>{team}</option>
              ))}
            </select>
            <span className="team-tag">LOCAL</span>
          </div>

          {/* VS Divider */}
          <div className="vs-divider">
            <span className="vs-text">VS</span>
          </div>

          {/* Away Team */}
          <div className="team-side away-side">
            <div
              className="team-badge"
              style={{
                background: formData.visitante
                  ? `linear-gradient(135deg, ${TEAM_COLORS[formData.visitante]}33, ${TEAM_COLORS[formData.visitante]}11)`
                  : 'var(--bg-elevated)',
                borderColor: formData.visitante
                  ? `${TEAM_COLORS[formData.visitante]}66`
                  : 'var(--border-light)',
              }}
            >
              <span className="badge-text">
                {formData.visitante ? TEAM_SHORT[formData.visitante] : '?'}
              </span>
            </div>
            <select
              name="visitante"
              value={formData.visitante}
              onChange={handleChange}
              className="team-select"
              required
            >
              <option value="">Equipo visitante</option>
              {awayTeams.map((team) => (
                <option key={team} value={team}>{team}</option>
              ))}
            </select>
            <span className="team-tag">VISITANTE</span>
          </div>
        </div>
      </div>

      {/* Odds Section - Collapsible */}
      <div className="odds-panel">
        <button
          type="button"
          className="odds-toggle"
          onClick={() => setShowOdds(!showOdds)}
        >
          <span className="odds-toggle-label">
            <span className="odds-icon">$</span>
            Cuotas de apuestas
          </span>
          <span className={`odds-arrow ${showOdds ? 'open' : ''}`}>
            &#9662;
          </span>
        </button>

        {showOdds && (
          <div className="odds-content">
            <div className="odds-inputs">
              <div className="odds-input-group">
                <label>1 (Local)</label>
                <input
                  type="number"
                  name="cuota_h"
                  step="0.01"
                  min="1.01"
                  value={formData.cuota_h}
                  onChange={handleChange}
                  className="odds-input"
                />
                {impliedProbs && (
                  <span className="implied-prob">{impliedProbs.home}%</span>
                )}
              </div>

              <div className="odds-input-group">
                <label>X (Empate)</label>
                <input
                  type="number"
                  name="cuota_d"
                  step="0.01"
                  min="1.01"
                  value={formData.cuota_d}
                  onChange={handleChange}
                  className="odds-input"
                />
                {impliedProbs && (
                  <span className="implied-prob">{impliedProbs.draw}%</span>
                )}
              </div>

              <div className="odds-input-group">
                <label>2 (Visitante)</label>
                <input
                  type="number"
                  name="cuota_a"
                  step="0.01"
                  min="1.01"
                  value={formData.cuota_a}
                  onChange={handleChange}
                  className="odds-input"
                />
                {impliedProbs && (
                  <span className="implied-prob">{impliedProbs.away}%</span>
                )}
              </div>
            </div>

            {impliedProbs && (
              <div className="odds-summary">
                <span>Margen casa: {impliedProbs.margin}%</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Validation error */}
      {validationError && (
        <div className="form-validation-error">{validationError}</div>
      )}

      {/* Submit */}
      <button
        type="submit"
        className="btn-predict"
        disabled={loading || !formData.local || !formData.visitante}
      >
        {loading ? (
          <span className="btn-loading">
            <span className="btn-spinner" />
            Analizando...
          </span>
        ) : (
          'Predecir resultado'
        )}
      </button>
    </form>
  )
}

export default PredictionForm
