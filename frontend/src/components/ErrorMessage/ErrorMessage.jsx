import './ErrorMessage.css'

export const ErrorMessage = ({ message, onDismiss }) => {
  return (
    <div className="error-message">
      <span className="error-icon">⚠️</span>
      <div className="error-content">
        <p>{message}</p>
      </div>
      {onDismiss && (
        <button className="error-close" onClick={onDismiss}>
          ✕
        </button>
      )}
    </div>
  )
}

export default ErrorMessage
