import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [message, setMessage] = useState('Loading...')
  const [name, setName] = useState('')
  const [personalizedMessage, setPersonalizedMessage] = useState('')

  useEffect(() => {
    // Fetch hello world message from backend
    fetch('http://localhost:8000/')
      .then(res => res.json())
      .then(data => setMessage(data.message))
      .catch(err => setMessage('Error connecting to backend'))
  }, [])

  const handleSubmit = (e) => {
    e.preventDefault()
    if (name.trim()) {
      fetch(`http://localhost:8000/api/v1/hello/${name}`)
        .then(res => res.json())
        .then(data => setPersonalizedMessage(data.message))
        .catch(err => setPersonalizedMessage('Error fetching message'))
    }
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>4dt907 - ML Data-Intensive System</h1>
        <p className="message">{message}</p>
        
        <div className="interactive-section">
          <h2>Try the API</h2>
          <form onSubmit={handleSubmit}>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Enter your name"
              className="name-input"
            />
            <button type="submit">Say Hello</button>
          </form>
          {personalizedMessage && (
            <p className="personalized-message">{personalizedMessage}</p>
          )}
        </div>

        <div className="info-section">
          <h3>Project Information</h3>
          <ul>
            <li>FastAPI Backend</li>
            <li>React Frontend</li>
            <li>MLflow Integration</li>
            <li>Docker Compose Orchestration</li>
          </ul>
        </div>
      </header>
    </div>
  )
}

export default App
