import './Loading.css'

export const Loading = ({ message = 'Cargando...' }) => {
  return (
    <div className="loading-container">
      <div className="spinner"></div>
      <p>{message}</p>
    </div>
  )
}

export default Loading
