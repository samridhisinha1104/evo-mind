import { useState, useEffect, useRef, useCallback } from 'react'
import './App.css'

const API_BASE = ''

const AVAILABLE_STEPS = [
  'describe_data', 'missing_value_report', 'correlation_analysis',
  'outlier_detection_iqr', 'distribution_analysis', 'kmeans_clustering',
  'pca_projection', 'categorical_frequency', 'data_profiling',
  'feature_importance_rf', 'anomaly_detection_iforest', 'mutual_information',
  'chi_squared_test', 'group_comparison_ttest', 'dbscan_clustering',
  'time_series_decomposition',
]

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
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('evomind-theme') ||
      (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
  })
  const wsRef = useRef(null)

  // Theme
  useEffect(() => {
    localStorage.setItem('evomind-theme', theme)
    document.documentElement.dataset.theme = theme
  }, [theme])

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
      } catch {
        setApiConnected(false)
      }
    }
    fetchJobs()
    const interval = setInterval(fetchJobs, 3000)
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
      if (res.ok) setMemory(await res.json())
    } catch { /* ignore */ }
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
      setTimeout(async () => {
        try {
          const jobRes = await fetch(`${API_BASE}/api/jobs/${data.job_id}`)
          if (jobRes.ok) setCurrentJob(await jobRes.json())
        } catch { /* ignore */ }
      }, 500)
      setView('job')
      setSidebarOpen(false)
      setFile(null)
      setTask('')
    } catch {
      alert('Failed to submit. Is the backend server running?')
    }
    setIsSubmitting(false)
  }

  const viewJob = (job) => {
    setCurrentJob(job)
    if (job.status === 'running') connectWs(job.id)
    setView('job')
    setSidebarOpen(false)
  }

  const navigate = (v) => {
    setView(v)
    setSidebarOpen(false)
  }

  return (
    <div className="app-shell" data-theme={theme}>
      {/* Mobile overlay backdrop */}
      {sidebarOpen && (
        <div className="sidebar-backdrop" onClick={() => setSidebarOpen(false)} />
      )}

      {/* Sidebar */}
      <aside className={`sidebar${sidebarOpen ? ' sidebar--open' : ''}`}>
        <div className="brand">
          <span className="brand-mark">🧬</span>
          <span className="brand-name">EvoMind</span>
          <span className="brand-version">v0.2</span>
        </div>

        <p className="nav-label">WORKSPACE</p>
        <nav className="nav-links">
          <button className={view === 'dashboard' ? 'nav-item active' : 'nav-item'} onClick={() => navigate('dashboard')}>
            <span className="nav-icon">📊</span> Dashboard
          </button>
          <button className={view === 'submit' ? 'nav-item active' : 'nav-item'} onClick={() => navigate('submit')}>
            <span className="nav-icon">🚀</span> New Analysis
          </button>
          <button className={view === 'memory' ? 'nav-item active' : 'nav-item'} onClick={() => { navigate('memory'); fetchMemory() }}>
            <span className="nav-icon">🧠</span> Memory
          </button>
          <button className={view === 'job' ? 'nav-item active' : 'nav-item'} onClick={() => navigate('job')} disabled={!currentJob}>
            <span className="nav-icon">⚡</span> Live Run
            {currentJob && currentJob.status === 'running' && <span className="live-dot" />}
          </button>
        </nav>

        <div className="sidebar-bottom">
          <div className="status-row">
            <div className={`status-dot ${apiConnected ? 'connected' : ''}`} />
            <span>Backend {apiConnected ? 'Online' : 'Offline'}</span>
          </div>
          {currentJob && currentJob.status === 'running' && (
            <div className="status-row">
              <div className={`status-dot ${wsConnected ? 'connected' : ''}`} />
              <span>WebSocket {wsConnected ? 'Live' : 'Disconnected'}</span>
            </div>
          )}
          <button className="theme-toggle" onClick={() => setTheme(t => t === 'light' ? 'dark' : 'light')}>
            {theme === 'light' ? '🌙' : '☀️'} {theme === 'light' ? 'Dark' : 'Light'} mode
          </button>
        </div>
      </aside>

      {/* Main */}
      <main className="main-content">
        <header className="topbar">
          {/* Hamburger — mobile only */}
          <button
            className="hamburger"
            aria-label="Toggle navigation"
            onClick={() => setSidebarOpen(s => !s)}
          >
            <span /><span /><span />
          </button>

          <div className="breadcrumbs">
            <span>EvoMind</span>
            <span className="sep">/</span>
            <strong>{
              view === 'dashboard' ? 'Dashboard'
              : view === 'submit' ? 'New Analysis'
              : view === 'memory' ? 'Strategy Memory'
              : 'Live Run'
            }</strong>
          </div>
        </header>

        {view === 'dashboard' && <DashboardView jobs={jobs} onSelectJob={viewJob} onNewAnalysis={() => navigate('submit')} />}
        {view === 'submit' && <SubmitView file={file} setFile={setFile} task={task} setTask={setTask} iterations={iterations} setIterations={setIterations} threshold={threshold} setThreshold={setThreshold} onSubmit={handleSubmit} isSubmitting={isSubmitting} onCancel={() => navigate('dashboard')} />}
        {view === 'memory' && <MemoryView memory={memory} />}
        {view === 'job' && <JobView job={currentJob} wsConnected={wsConnected} />}
      </main>
    </div>
  )
}


// ============================================================
// Dashboard
// ============================================================

function DashboardView({ jobs, onSelectJob, onNewAnalysis }) {
  const running = jobs.filter(j => j.status === 'running')
  const completed = jobs.filter(j => j.status === 'completed')
  const failed = jobs.filter(j => j.status === 'failed')
  const bestScore = completed.length > 0
    ? Math.max(...completed.map(j => j.result?.best_score || 0))
    : null

  return (
    <div className="view fade-in">
      <section className="page-heading">
        <div>
          <p className="eyebrow">YOUR WORKSPACE</p>
          <h1>What's in your data?</h1>
          <p className="subtitle">Drop in a dataset, tell EvoMind what you're looking for, and it'll try different approaches until it finds something useful.</p>
        </div>
        <button className="primary-button" onClick={onNewAnalysis}>
          + New analysis
        </button>
      </section>

      <div className="stats-grid">
        <article className="stat-card">
          <div className="stat-head"><span>Total runs</span><span>📊</span></div>
          <strong>{jobs.length}</strong>
          <small>{running.length} active</small>
        </article>
        <article className="stat-card accent-blue">
          <div className="stat-head"><span>Running</span><span>⚡</span></div>
          <strong>{running.length}</strong>
          <small>{completed.length} completed</small>
        </article>
        <article className="stat-card accent-green">
          <div className="stat-head"><span>Best score</span><span>🏆</span></div>
          <strong>{bestScore !== null ? bestScore.toFixed(4) : '—'}</strong>
          <small>{failed.length} failed</small>
        </article>
      </div>

      {/* Recent jobs */}
      <section className="panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">RECENT ACTIVITY</p>
            <h2>Your analyses</h2>
          </div>
        </div>

        {jobs.length === 0 ? (
          <div className="empty-state">
            <p className="empty-icon">🧬</p>
            <p><strong>Nothing here yet</strong></p>
            <p>Upload a dataset and describe what you want to find — EvoMind will do the rest.</p>
            <button className="primary-button small" onClick={onNewAnalysis}>+ New analysis</button>
          </div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Task</th>
                  <th>Dataset</th>
                  <th>Status</th>
                  <th>Iterations</th>
                  <th>Best Score</th>
                </tr>
              </thead>
              <tbody>
                {jobs.slice().reverse().map(job => (
                  <tr key={job.id} className="clickable-row" onClick={() => onSelectJob(job)}>
                    <td><code>#{job.id}</code></td>
                    <td className="task-cell">{job.task}</td>
                    <td>{job.filename || '—'}</td>
                    <td>
                      <span className={`status-pill ${job.status}`}>
                        <i /> {job.status}
                      </span>
                    </td>
                    <td>{job.history?.length || 0} / {job.iterations}</td>
                    <td className="mono">{job.result?.best_score?.toFixed(4) || (job.history?.length > 0 ? Math.max(...job.history.map(h => h.best_score || 0)).toFixed(4) : '—')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  )
}


// ============================================================
// New Analysis Form
// ============================================================

function SubmitView({ file, setFile, task, setTask, iterations, setIterations, threshold, setThreshold, onSubmit, isSubmitting, onCancel }) {
  const [showSteps, setShowSteps] = useState(false)

  return (
    <div className="view fade-in">
      <section className="page-heading">
        <div>
          <p className="eyebrow">NEW ANALYSIS</p>
          <h1>Start a new analysis</h1>
          <p className="subtitle">Upload a dataset and describe what you're trying to find. EvoMind will run different approaches and keep what works best.</p>
        </div>
      </section>

      {/* ── Disclaimer banner ── */}
      <div className="disclaimer-banner" role="note" aria-label="How EvoMind works">
        <div className="disclaimer-icon">💡</div>
        <div className="disclaimer-body">
          <strong>EvoMind explores your data — it doesn't answer questions directly.</strong>
          <p>
            It runs analysis techniques like correlation, clustering, and outlier detection across your dataset,
            then keeps refining the approach until it finds something meaningful. Think of it as a data exploration
            tool, not a search engine.
          </p>
          <div className="disclaimer-examples">
            <div className="example bad">
              <span className="example-label">✗ Try rephrasing this</span>
              <span className="example-text">"Which country got the most gold medals?"</span>
            </div>
            <div className="example good">
              <span className="example-label">✓ This works well</span>
              <span className="example-text">"Find patterns in how medals are distributed across countries and highlight what's driving the differences."</span>
            </div>
          </div>
          <button className="steps-toggle" onClick={() => setShowSteps(s => !s)}>
            {showSteps ? '▲ Hide' : '▼ Show'} what EvoMind can analyse ({AVAILABLE_STEPS.length} techniques)
          </button>
          {showSteps && (
            <div className="steps-grid">
              {AVAILABLE_STEPS.map(s => (
                <span key={s} className="step-chip">{s.replace(/_/g, ' ')}</span>
              ))}
            </div>
          )}
        </div>
      </div>

      <form className="panel form-panel" onSubmit={onSubmit}>
        <div className="form-group">
          <label>Dataset</label>
          <label className="dropzone" htmlFor="file-input">
            <input id="file-input" type="file" accept=".csv,.parquet,.json,.jsonl,.xlsx,.tsv" onChange={(e) => setFile(e.target.files[0])} />
            {file ? (
              <div className="file-selected">
                <span>📄</span>
                <strong>{file.name}</strong>
                <small>({(file.size / 1024).toFixed(1)} KB)</small>
              </div>
            ) : (
              <>
                <span className="drop-icon">📁</span>
                <strong>Click to select a file</strong>
                <small>CSV, Parquet, JSON, JSONL, Excel, TSV</small>
              </>
            )}
          </label>
        </div>

        <div className="form-group">
          <label htmlFor="task-input">Analysis Goal</label>
          <textarea
            id="task-input"
            value={task}
            onChange={(e) => setTask(e.target.value)}
            placeholder="e.g. Look for patterns in sales across regions. Highlight which factors seem to drive revenue and flag anything unusual."
            rows={4}
          />
          <small className="form-hint">What do you want to know about your data? Plain language is fine.</small>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label htmlFor="iter-input">Attempts</label>
            <input id="iter-input" type="number" value={iterations} onChange={(e) => setIterations(parseInt(e.target.value) || 1)} min={1} max={20} />
            <small className="form-hint">How many approaches to try before stopping (1–20). More attempts = more thorough.</small>
          </div>
          <div className="form-group">
            <label htmlFor="thresh-input">Stop when score reaches</label>
            <input id="thresh-input" type="number" value={threshold} onChange={(e) => setThreshold(parseFloat(e.target.value) || 0)} min={0} max={1} step={0.05} />
            <small className="form-hint">Stops early if results look good enough (0 = never stop early, 1 = only perfect).</small>
          </div>
        </div>

        <div className="form-actions">
          <button type="button" className="quiet-button" onClick={onCancel}>Cancel</button>
          <button type="submit" className="primary-button" disabled={!file || !task || isSubmitting}>
            {isSubmitting ? '⏳ Starting...' : '🧬 Run analysis'}
          </button>
        </div>
      </form>
    </div>
  )
}


// ============================================================
// Live Run
// ============================================================

function MultiObjBadges({ evaluation }) {
  if (!evaluation) return null
  const dims = [
    { key: 'insight_depth', label: 'Depth', color: 'blue' },
    { key: 'coverage', label: 'Breadth', color: 'green' },
    { key: 'novelty', label: 'Fresh', color: 'yellow' },
    { key: 'efficiency', label: 'Clean', color: 'muted' },
  ]
  return (
    <div className="multiobj-badges">
      {dims.map(({ key, label, color }) => (
        evaluation[key] != null && (
          <span key={key} className={`multiobj-badge multiobj-badge--${color}`} title={`${label}: ${evaluation[key]}`}>
            {label} {(evaluation[key] * 100).toFixed(0)}%
          </span>
        )
      ))}
    </div>
  )
}

function JobView({ job, wsConnected }) {
  const [expandedRows, setExpandedRows] = useState({})

  if (!job) {
    return (
      <div className="view fade-in">
        <div className="empty-state">
          <p className="empty-icon">⚡</p>
          <p><strong>Nothing running yet</strong></p>
          <p>Start a new analysis to see results here as they come in.</p>
        </div>
      </div>
    )
  }

  const history = job.history || []
  const maxScore = history.length > 0
    ? Math.max(...history.map(h => h.evaluation?.score || h.best_score || 0))
    : 0
  const progress = job.iterations > 0 ? (history.length / job.iterations) * 100 : 0

  const toggleRow = (i) => setExpandedRows(prev => ({ ...prev, [i]: !prev[i] }))

  return (
    <div className="view fade-in">
      <section className="page-heading">
        <div>
          <p className="eyebrow">IN PROGRESS</p>
          <h1>Run #{job.id}</h1>
          <p className="subtitle">{job.task}</p>
        </div>
        <span className={`status-pill large ${job.status}`}>
          <i /> {job.status}
        </span>
      </section>

      {/* Info bar */}
      <div className="job-info-bar">
        <span>📄 {job.filename}</span>
        <span>🔄 {history.length} / {job.iterations} iterations</span>
        <span>🎯 Threshold: {job.threshold}</span>
        {wsConnected && job.status === 'running' && <span className="ws-badge">● Live</span>}
      </div>

      {/* Progress */}
      {job.status === 'running' && (
        <div className="progress-line">
          <span style={{ width: `${progress}%` }} />
        </div>
      )}

      {/* Running indicator */}
      {job.status === 'running' && history.length === 0 && (
        <div className="ai-working-banner">
          <span className="ai-spinner">🧬</span>
          <div>
            <strong>Getting started…</strong>
            <p>EvoMind is reading your dataset and planning its first approach. Hang tight — the first result will show up shortly.</p>
          </div>
        </div>
      )}

      {job.status === 'running' && history.length > 0 && (
        <div className="ai-working-banner ai-working-banner--running">
          <span className="ai-spinner">⚡</span>
          <div>
            <strong>Trying approach {history.length} of {job.iterations}…</strong>
            <p>Tweaking the previous approach and checking if the new one does better.</p>
          </div>
        </div>
      )}

      {/* Score chart */}
      <section className="panel chart-panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">PROGRESS</p>
            <h2>Score per attempt</h2>
          </div>
          {history.length > 0 && (
            <strong className="chart-best">Best: {maxScore.toFixed(4)}</strong>
          )}
        </div>
        <div className="chart-scroll-wrap">
          <div className="chart">
            {history.length === 0 ? (
              <div className="empty-chart">Waiting for first result…</div>
            ) : (
              history.map((h, i) => {
                const score = h.evaluation?.score || h.best_score || 0
                const height = Math.max(score * 100, 3)
                const isBest = score === maxScore
                return (
                  <div className="bar-wrap" key={i}>
                    <div className={`bar ${isBest ? 'bar-best' : ''}`} style={{ height: `${height}%` }} title={`Iter ${i}: ${score.toFixed(4)}`}>
                      <span className="bar-label">{score.toFixed(2)}</span>
                    </div>
                    <span>#{i}</span>
                  </div>
                )
              })
            )}
          </div>
        </div>
      </section>

      {/* Evolution history table */}
      {history.length > 0 && (
        <section className="panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">ATTEMPT LOG</p>
              <h2>What was tried</h2>
            </div>
            <p className="panel-hint">Click any row to see why it scored the way it did</p>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Approach</th>
                  <th>Change made</th>
                  <th>Gen</th>
                  <th>Score</th>
                  <th>Quality breakdown</th>
                  <th>What it ran</th>
                </tr>
              </thead>
              <tbody>
                {history.map((h, i) => {
                  const score = h.evaluation?.score || h.best_score || 0
                  const isBestRow = score === maxScore
                  const isExpanded = expandedRows[i]
                  return (
                    <>
                      <tr
                        key={`row-${i}`}
                        className={`clickable-row${isBestRow && i === history.length - 1 ? ' best-row' : ''}`}
                        onClick={() => toggleRow(i)}
                        title="Click to see why this scored the way it did"
                      >
                        <td>{h.iteration ?? i}</td>
                        <td className="mono">{h.strategy?.name || '—'}</td>
                        <td><span className="mutation-badge">{h.strategy?.mutation_applied || 'initial'}</span></td>
                        <td>{h.strategy?.generation ?? 0}</td>
                        <td className="score-cell mono">{score.toFixed(4)}</td>
                        <td>
                          <MultiObjBadges evaluation={h.evaluation} />
                        </td>
                        <td className="steps-cell">{h.strategy?.steps?.join(' → ') || '—'}</td>
                      </tr>
                      {isExpanded && h.evaluation?.rationale && (
                        <tr key={`rationale-${i}`} className="rationale-row">
                          <td colSpan={7}>
                            <div className="rationale-box">
                              <span className="rationale-label">💬 Why this score</span>
                              <p>{h.evaluation.rationale}</p>
                              {h.evaluation.signal_count != null && (
                                <span className="signal-count">
                                  Interesting findings: <strong>{h.evaluation.signal_count}</strong>
                                </span>
                              )}
                            </div>
                          </td>
                        </tr>
                      )}
                    </>
                  )
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* Final result */}
      {job.result && (
        <section className="panel result-card">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">DONE</p>
              <h2>🏆 {job.result.best_strategy?.name || 'Best approach'}</h2>
            </div>
          </div>
          <div className="result-body">
            <div className="result-score">{job.result.best_score?.toFixed(4)}</div>
            <div className="result-meta">
              <p><strong>Finished because:</strong> {job.result.stop_reason === 'score_threshold_reached' ? '✅ Hit the target score' : job.result.stop_reason === 'max_iterations_reached' ? '🔁 Ran all attempts' : job.result.stop_reason}</p>
              <p><strong>Attempts made:</strong> {job.result.total_iterations}</p>
              <p><strong>Techniques used:</strong> {job.result.best_strategy?.steps?.join(' → ')}</p>
              <p><strong>Generation:</strong> {job.result.best_strategy?.generation}</p>
              {job.result.best_strategy?.params && (
                <p><strong>Params:</strong> <code>{JSON.stringify(job.result.best_strategy.params)}</code></p>
              )}
              {job.result.dataset_summary && (
                <p><strong>Dataset:</strong> {job.result.dataset_summary.n_rows?.toLocaleString()} rows × {job.result.dataset_summary.n_cols} cols</p>
              )}
            </div>
          </div>
        </section>
      )}

      {/* Error */}
      {job.error && (
        <section className="panel error-card">
          <p className="eyebrow">SOMETHING WENT WRONG</p>
          <p>{job.error}</p>
        </section>
      )}
    </div>
  )
}


// ============================================================
// Memory
// ============================================================

function MemoryView({ memory }) {
  if (!memory) {
    return (
      <div className="view fade-in">
        <div className="empty-state">
          <p className="empty-icon">🧠</p>
          <p><strong>Loading…</strong></p>
        </div>
      </div>
    )
  }

  const strategies = memory.global_best || []

  return (
    <div className="view fade-in">
      <section className="page-heading">
        <div>
          <p className="eyebrow">WHAT'S BEEN LEARNED</p>
          <h1>Saved approaches</h1>
          <p className="subtitle">The best approaches from all your previous runs, ranked by how well they worked. EvoMind reuses these as a starting point for similar tasks.</p>
        </div>
      </section>

      {strategies.length === 0 ? (
        <div className="empty-state">
          <p className="empty-icon">🧠</p>
          <p><strong>Nothing saved yet</strong></p>
          <p>Once you run an analysis, the best approaches get saved here so future runs can build on them.</p>
        </div>
      ) : (
        <section className="panel">
          {strategies.map((s, i) => (
            <article className="memory-row" key={i}>
              <span className="memory-rank">#{i + 1}</span>
              <div className="memory-info">
                <h3>{s.name}</h3>
                <p className="memory-sig">Task: {s.task_signature}</p>
                <div className="memory-steps">
                  {s.strategy?.steps?.map((step, j) => (
                    <span key={j} className="step-tag">{step}</span>
                  ))}
                </div>
              </div>
              <div className="memory-score">{s.score?.toFixed(4)}</div>
            </article>
          ))}
        </section>
      )}
    </div>
  )
}


export default App