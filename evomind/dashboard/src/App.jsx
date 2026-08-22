import { useState, useEffect, useRef, useCallback } from 'react'
import './App.css'

const API_BASE = ''

function App() {
  const [view, setView] = useState('dashboard')
  const [jobs, setJobs] = useState([])
  const [currentJob, setCurrentJob] = useState(null)
  const [memory, setMemory] = useState(null)
  const [file, setFile] = useState(null)
  const [task, setTask] = useState('')
  const [iterations, setIterations] = useState(5)
  const [threshold, setThreshold] = useState(0.8)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [wsConnected, setWsConnected] = useState(false)
  const [apiConnected, setApiConnected] = useState(false)
  const wsRef = useRef(null)

  // Fetch jobs
  useEffect(() => {
    const fetchJobs = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/jobs`)
        if (res.ok) {
          const data = await res.json()
          setJobs(data)
          setApiConnected(true)
        } else {
          setApiConnected(false)
        }
      } catch (e) {
        setApiConnected(false)
      }
    }
    fetchJobs()
    const interval = setInterval(fetchJobs, 2000)
    return () => clearInterval(interval)
  }, [])

  // WebSocket connection for current job
  const connectWs = useCallback((jobId) => {
    if (wsRef.current) wsRef.current.close()
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsHost = window.location.host
    const ws = new WebSocket(`${wsProtocol}//${wsHost}/ws/jobs/${jobId}`)
    ws.onopen = () => setWsConnected(true)
    ws.onclose = () => setWsConnected(false)
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      setCurrentJob(data)
    }
    wsRef.current = ws
  }, [])

  // Fetch memory
  const fetchMemory = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/memory`)
      if (res.ok) {
        setMemory(await res.json())
        setApiConnected(true)
      } else {
        setApiConnected(false)
      }
    } catch (e) {
      setApiConnected(false)
    }
  }

  // Submit analysis
  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!file || !task) return
    setIsSubmitting(true)

    const formData = new FormData()
    formData.append('file', file)
    formData.append('task', task)
    formData.append('iterations', iterations)
    formData.append('threshold', threshold)

    try {
      const res = await fetch(`${API_BASE}/api/analyze`, { method: 'POST', body: formData })
      const data = await res.json()
      connectWs(data.job_id)
      setView('job')
      // Poll for initial state
      setTimeout(async () => {
        const jobRes = await fetch(`${API_BASE}/api/jobs/${data.job_id}`)
        setCurrentJob(await jobRes.json())
      }, 500)
    } catch (e) {
      alert('Failed to submit. Is the server running?')
    }
    setIsSubmitting(false)
  }

  return (
    <div className="app">
      <nav className="sidebar">
        <div className="logo">
          <span className="logo-icon">🧬</span>
          <h1>EvoMind</h1>
          <span className="version">v0.2</span>
        </div>
        <div className="nav-links">
          <button className={view === 'dashboard' ? 'active' : ''} onClick={() => setView('dashboard')}>
            <span className="nav-icon">📊</span> Dashboard
          </button>
          <button className={view === 'submit' ? 'active' : ''} onClick={() => setView('submit')}>
            <span className="nav-icon">🚀</span> New Analysis
          </button>
          <button className={view === 'memory' ? 'active' : ''} onClick={() => { setView('memory'); fetchMemory() }}>
            <span className="nav-icon">🧠</span> Memory
          </button>
          <button className={view === 'job' ? 'active' : ''} onClick={() => setView('job')} disabled={!currentJob}>
            <span className="nav-icon">⚡</span> Live Run
          </button>
        </div>
        <div className="sidebar-footer">
          <div className="status-item">
            <div className={`status-dot ${apiConnected ? 'connected' : ''}`} />
            <span>Backend: {apiConnected ? 'Online' : 'Offline'}</span>
          </div>
          {currentJob && (
            <div className="status-item">
              <div className={`status-dot ${wsConnected ? 'connected' : ''}`} />
              <span>Live Run: {wsConnected ? 'Connected' : 'Disconnected'}</span>
            </div>
          )}
        </div>
      </nav>

      <main className="content">
        {view === 'dashboard' && <DashboardView jobs={jobs} onSelectJob={(job) => { setCurrentJob(job); setView('job') }} />}
        {view === 'submit' && (
          <SubmitView
            file={file} setFile={setFile}
            task={task} setTask={setTask}
            iterations={iterations} setIterations={setIterations}
            threshold={threshold} setThreshold={setThreshold}
            onSubmit={handleSubmit} isSubmitting={isSubmitting}
          />
        )}
        {view === 'memory' && <MemoryView memory={memory} />}
        {view === 'job' && <JobView job={currentJob} />}
      </main>
    </div>
  )
}


function DashboardView({ jobs, onSelectJob }) {
  const running = jobs.filter(j => j.status === 'running')
  const completed = jobs.filter(j => j.status === 'completed')
  const failed = jobs.filter(j => j.status === 'failed')

  return (
    <div className="view">
      <h2 className="view-title">Dashboard</h2>
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-value">{jobs.length}</div>
          <div className="stat-label">Total Runs</div>
        </div>
        <div className="stat-card running">
          <div className="stat-value">{running.length}</div>
          <div className="stat-label">Running</div>
        </div>
        <div className="stat-card success">
          <div className="stat-value">{completed.length}</div>
          <div className="stat-label">Completed</div>
        </div>
        <div className="stat-card error">
          <div className="stat-value">{failed.length}</div>
          <div className="stat-label">Failed</div>
        </div>
      </div>

      <h3 className="section-title">Recent Jobs</h3>
      <div className="job-list">
        {jobs.length === 0 ? (
          <div className="empty-state">
            <p>No jobs yet. Submit a new analysis to get started!</p>
          </div>
        ) : (
          jobs.map(job => (
            <div key={job.id} className={`job-card ${job.status}`} onClick={() => onSelectJob(job)}>
              <div className="job-header">
                <span className="job-id">#{job.id}</span>
                <span className={`badge ${job.status}`}>{job.status}</span>
              </div>
              <div className="job-task">{job.task}</div>
              <div className="job-meta">
                <span>{job.filename}</span>
                <span>{job.history?.length || 0} iterations</span>
                {job.result && <span>Score: {job.result.best_score?.toFixed(4)}</span>}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}


function SubmitView({ file, setFile, task, setTask, iterations, setIterations, threshold, setThreshold, onSubmit, isSubmitting }) {
  return (
    <div className="view">
      <h2 className="view-title">🚀 New Analysis</h2>
      <form className="submit-form" onSubmit={onSubmit}>
        <div className="form-group">
          <label>Dataset</label>
          <div className="file-drop" onClick={() => document.getElementById('file-input').click()}>
            <input id="file-input" type="file" accept=".csv,.parquet,.json,.jsonl,.xlsx,.tsv" onChange={(e) => setFile(e.target.files[0])} />
            {file ? (
              <div className="file-selected">
                <span className="file-icon">📄</span>
                <span>{file.name}</span>
                <span className="file-size">({(file.size / 1024).toFixed(1)} KB)</span>
              </div>
            ) : (
              <div className="file-placeholder">
                <span className="drop-icon">📁</span>
                <p>Click to select a file</p>
                <span className="formats">CSV, Parquet, JSON, JSONL, Excel, TSV</span>
              </div>
            )}
          </div>
        </div>

        <div className="form-group">
          <label>Task Description</label>
          <textarea
            value={task}
            onChange={(e) => setTask(e.target.value)}
            placeholder="Analyze this dataset and discover useful patterns..."
            rows={3}
          />
        </div>

        <div className="form-row">
          <div className="form-group">
            <label>Max Iterations</label>
            <input type="number" value={iterations} onChange={(e) => setIterations(parseInt(e.target.value))} min={1} max={20} />
          </div>
          <div className="form-group">
            <label>Score Threshold</label>
            <input type="number" value={threshold} onChange={(e) => setThreshold(parseFloat(e.target.value))} min={0} max={1} step={0.05} />
          </div>
        </div>

        <button type="submit" className="btn-primary" disabled={!file || !task || isSubmitting}>
          {isSubmitting ? 'Submitting...' : '🧬 Start Evolution'}
        </button>
      </form>
    </div>
  )
}


function JobView({ job }) {
  if (!job) return <div className="view"><div className="empty-state"><p>No active job. Submit an analysis first.</p></div></div>

  const history = job.history || []
  const maxScore = Math.max(...history.map(h => h.evaluation?.score || h.best_score || 0), 0)

  return (
    <div className="view">
      <div className="job-view-header">
        <h2 className="view-title">⚡ Live Run: #{job.id}</h2>
        <span className={`badge large ${job.status}`}>{job.status}</span>
      </div>

      <div className="job-info-bar">
        <span>📋 {job.task}</span>
        <span>📄 {job.filename}</span>
      </div>

      {/* Score chart */}
      <div className="chart-card">
        <h3>Score Trajectory</h3>
        <div className="score-chart">
          {history.map((h, i) => {
            const score = h.evaluation?.score || h.best_score || 0
            const height = Math.max(score * 100, 2)
            return (
              <div key={i} className="chart-bar-wrapper">
                <div className="chart-bar" style={{ height: `${height}%` }} title={`Iter ${i}: ${score.toFixed(4)}`}>
                  <span className="bar-label">{score.toFixed(2)}</span>
                </div>
                <span className="bar-iter">#{i}</span>
              </div>
            )
          })}
          {history.length === 0 && <div className="empty-chart">Waiting for first iteration...</div>}
        </div>
        {job.threshold && (
          <div className="threshold-line" style={{ bottom: `${job.threshold * 100}%` }}>
            Threshold: {job.threshold}
          </div>
        )}
      </div>

      {/* Evolution history table */}
      <div className="table-card">
        <h3>Evolution History</h3>
        <table>
          <thead>
            <tr>
              <th>Iter</th>
              <th>Strategy</th>
              <th>Mutation</th>
              <th>Gen</th>
              <th>Score</th>
              <th>Steps</th>
            </tr>
          </thead>
          <tbody>
            {history.map((h, i) => (
              <tr key={i} className={h.best_score === maxScore && i === history.length - 1 ? 'best-row' : ''}>
                <td>{h.iteration ?? i}</td>
                <td className="strategy-name">{h.strategy?.name || 'N/A'}</td>
                <td><span className="mutation-badge">{h.strategy?.mutation_applied || 'initial'}</span></td>
                <td>{h.strategy?.generation ?? 0}</td>
                <td className="score-cell">{(h.evaluation?.score || h.best_score || 0).toFixed(4)}</td>
                <td className="steps-cell">{h.strategy?.steps?.join(', ') || 'N/A'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Result */}
      {job.result && (
        <div className="result-card">
          <h3>🏆 Best Strategy: {job.result.best_strategy?.name}</h3>
          <div className="result-details">
            <div className="result-score">{job.result.best_score?.toFixed(4)}</div>
            <div className="result-meta">
              <p><strong>Stop reason:</strong> {job.result.stop_reason}</p>
              <p><strong>Steps:</strong> {job.result.best_strategy?.steps?.join(' → ')}</p>
              <p><strong>Params:</strong> <code>{JSON.stringify(job.result.best_strategy?.params || {})}</code></p>
              <p><strong>Generation:</strong> {job.result.best_strategy?.generation}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}


function MemoryView({ memory }) {
  if (!memory) return <div className="view"><div className="empty-state"><p>Loading memory...</p></div></div>

  const strategies = memory.global_best || []

  return (
    <div className="view">
      <h2 className="view-title">🧠 Strategy Memory</h2>
      <p className="view-subtitle">Best strategies across all runs, ranked by score</p>

      <div className="memory-grid">
        {strategies.length === 0 ? (
          <div className="empty-state"><p>No strategies in memory yet. Run an analysis first!</p></div>
        ) : (
          strategies.map((s, i) => (
            <div key={i} className="memory-card">
              <div className="memory-rank">#{i + 1}</div>
              <div className="memory-info">
                <h4>{s.name}</h4>
                <div className="memory-score">{s.score?.toFixed(4)}</div>
                <div className="memory-sig">Task: {s.task_signature}</div>
                <div className="memory-steps">
                  {s.strategy?.steps?.map((step, j) => (
                    <span key={j} className="step-tag">{step}</span>
                  ))}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}

export default App
