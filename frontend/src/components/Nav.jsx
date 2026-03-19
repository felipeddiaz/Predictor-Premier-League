import { NavLink } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { api } from '../services/api'

function useDarkMode() {
  const [dark, setDark] = useState(() => localStorage.getItem('theme') === 'dark')

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light')
    localStorage.setItem('theme', dark ? 'dark' : 'light')
  }, [dark])

  // Apply saved theme on first render
  useEffect(() => {
    const saved = localStorage.getItem('theme')
    if (saved === 'dark') {
      document.documentElement.setAttribute('data-theme', 'dark')
      setDark(true)
    }
  }, [])

  return [dark, () => setDark(d => !d)]
}

export default function Nav() {
  const [status, setStatus] = useState({ text: 'Conectando\u2026', cls: '' })
  const [dark, toggleDark] = useDarkMode()

  useEffect(() => {
    const check = async () => {
      try {
        const data = await api.health()
        if (data.status === 'healthy') {
          setStatus({ text: 'API Conectada', cls: 'grn' })
        } else {
          setStatus({ text: 'API Degradada', cls: 'amb' })
        }
      } catch {
        setStatus({ text: 'API Offline', cls: 'red' })
      }
    }
    check()
    const id = setInterval(check, 30000)
    return () => clearInterval(id)
  }, [])

  const pillStyle = {
    background: status.cls === 'grn' ? 'var(--grn-l)' : status.cls === 'amb' ? 'var(--amb-l)' : status.cls === 'red' ? 'var(--red-l)' : 'var(--bg)',
    borderColor: status.cls === 'grn' ? 'var(--grn-b)' : status.cls === 'amb' ? 'var(--amb-b)' : status.cls === 'red' ? 'var(--red-b)' : 'var(--border)',
    color: status.cls === 'grn' ? 'var(--grn)' : status.cls === 'amb' ? 'var(--amb)' : status.cls === 'red' ? 'var(--red)' : 'var(--ink3)',
  }

  const pulseStyle = {
    width: 6, height: 6, borderRadius: '50%',
    background: pillStyle.color,
    animation: status.cls === 'grn' ? 'pulse 2s ease infinite' : 'none',
  }

  return (
    <nav>
      <NavLink to="/" className="nav-logo">
        <div className="nav-logo-dot" />
        PL Predictor
      </NavLink>
      <NavLink to="/" className={({ isActive }) => `ntab${isActive ? ' active' : ''}`} end>
        Predictor
      </NavLink>
      <NavLink to="/historial" className={({ isActive }) => `ntab${isActive ? ' active' : ''}`}>
        Historial
      </NavLink>
      <div className="nav-r">
        <button className="dm-toggle" onClick={toggleDark} title={dark ? 'Modo claro' : 'Modo oscuro'}>
          {dark ? '\u2600\uFE0F' : '\uD83C\uDF19'}
        </button>
        <div className="nav-pill" style={pillStyle}>
          <div style={pulseStyle} />
          {status.text}
        </div>
      </div>
    </nav>
  )
}
