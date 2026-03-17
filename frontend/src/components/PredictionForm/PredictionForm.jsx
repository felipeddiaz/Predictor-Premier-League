import { useState } from 'react'
import './PredictionForm.css'

const TEAMS = [
  'Arsenal',
  'Aston Villa',
  'Bournemouth',
  'Brentford',
  'Brighton',
  'Chelsea',
  'Crystal Palace',
  'Everton',
  'Fulham',
  'Ipswich',
  'Leicester',
  'Liverpool',
  'Manchester City',
  'Manchester United',
  'Newcastle',
  'Nottingham',
  'Southampton',
  'Tottenham',
  'West Ham',
  'Wolves',
]

export const PredictionForm = ({ onSubmit, loading }) => {
  const [formData, setFormData] = useState({
    local: '',
    visitante: '',
    cuota_h: '2.0',
    cuota_d: '3.4',
    cuota_a: '3.2',
  })

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }))
  }

  const handleSubmit = (e) => {
    e.preventDefault()

    if (!formData.local || !formData.visitante) {
      alert('Selecciona ambos equipos')
      return
    }

    if (formData.local === formData.visitante) {
      alert('Los equipos no pueden ser iguales')
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

  return (
    <form className="prediction-form" onSubmit={handleSubmit}>
      <div className="form-row">
        <div className="form-group">
          <label htmlFor="local">Equipo Local</label>
          <select
            id="local"
            name="local"
            value={formData.local}
            onChange={handleChange}
            className="form-input"
            required
          >
            <option value="">-- Selecciona equipo local --</option>
            {TEAMS.map((team) => (
              <option key={team} value={team}>
                {team}
              </option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label htmlFor="visitante">Equipo Visitante</label>
          <select
            id="visitante"
            name="visitante"
            value={formData.visitante}
            onChange={handleChange}
            className="form-input"
            required
          >
            <option value="">-- Selecciona equipo visitante --</option>
            {TEAMS.map((team) => (
              <option key={team} value={team}>
                {team}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="cuotas-section">
        <h3>Cuotas de Apuestas (Opcional)</h3>
        <div className="cuotas-grid">
          <div className="form-group">
            <label htmlFor="cuota_h">Victoria Local</label>
            <input
              type="number"
              id="cuota_h"
              name="cuota_h"
              step="0.01"
              min="1"
              value={formData.cuota_h}
              onChange={handleChange}
              className="form-input"
            />
          </div>

          <div className="form-group">
            <label htmlFor="cuota_d">Empate</label>
            <input
              type="number"
              id="cuota_d"
              name="cuota_d"
              step="0.01"
              min="1"
              value={formData.cuota_d}
              onChange={handleChange}
              className="form-input"
            />
          </div>

          <div className="form-group">
            <label htmlFor="cuota_a">Victoria Visitante</label>
            <input
              type="number"
              id="cuota_a"
              name="cuota_a"
              step="0.01"
              min="1"
              value={formData.cuota_a}
              onChange={handleChange}
              className="form-input"
            />
          </div>
        </div>
      </div>

      <button
        type="submit"
        className="btn-submit"
        disabled={loading || !formData.local || !formData.visitante}
      >
        {loading ? 'Analizando...' : 'Obtener Predicción'}
      </button>
    </form>
  )
}

export default PredictionForm
