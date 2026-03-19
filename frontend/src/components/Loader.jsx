export default function Loader({ visible, text = 'Calculando\u2026' }) {
  return (
    <div className={`ldr${visible ? ' on' : ''}`}>
      <div className="ldr-ring" />
      <div className="ldr-txt">{text}</div>
    </div>
  )
}
