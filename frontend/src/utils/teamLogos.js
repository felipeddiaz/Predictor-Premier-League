import { useState } from 'react'

// football-data.org free SVG crests — no auth required
const FD_CDN = 'https://crests.football-data.org'

const TEAM_IDS = {
  'Arsenal':       57,
  'Aston Villa':   58,
  'Bournemouth':   1044,
  'Brentford':     402,
  'Brighton':      397,
  'Chelsea':       61,
  'Crystal Palace':354,
  'Everton':       62,
  'Fulham':        63,
  'Ipswich':       349,
  'Leicester':     338,
  'Liverpool':     64,
  'Man City':      65,
  'Man United':    66,
  'Newcastle':     67,
  "Nott'm Forest": 351,
  'Southampton':   340,
  'Tottenham':     73,
  'West Ham':      563,
  'Wolves':        76,
}

export function teamLogoUrl(name) {
  const id = TEAM_IDS[name]
  return id ? `${FD_CDN}/${id}.svg` : null
}

function initials(name) {
  return (name || '').split(' ').map(w => w[0]).join('').slice(0, 3).toUpperCase()
}

export function TeamBadge({ name, size = 22, style = {}, className = '' }) {
  const [err, setErr] = useState(false)
  const url = teamLogoUrl(name)

  const base = {
    width: size, height: size, flexShrink: 0,
    borderRadius: size > 30 ? 10 : 5,
    objectFit: 'contain',
    ...style,
  }

  if (url && !err) {
    return (
      <img
        src={url}
        alt={name}
        className={`team-logo ${className}`}
        style={base}
        onError={() => setErr(true)}
      />
    )
  }

  // Fallback: initials badge
  return (
    <div
      className={`mi-badge ${className}`}
      style={{ width: size, height: size, borderRadius: size > 30 ? 10 : 5, fontSize: size > 30 ? 11 : 8, ...style }}
    >
      {initials(name)}
    </div>
  )
}
