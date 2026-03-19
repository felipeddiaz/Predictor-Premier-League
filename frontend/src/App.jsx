import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Nav from './components/Nav'
import Predictor from './pages/Predictor'
import Historial from './pages/Historial'
import './styles/global.css'

export default function App() {
  return (
    <BrowserRouter>
      <Nav />
      <div className="shell">
        <Routes>
          <Route path="/" element={<Predictor />} />
          <Route path="/historial" element={<Historial />} />
        </Routes>
      </div>
    </BrowserRouter>
  )
}
