import { useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState('What are the most important trends in this data?');
  const [sheetName, setSheetName] = useState('');
  const [research, setResearch] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  async function submit(event) {
    event.preventDefault(); setError(''); setResult(null);
    if (!file) return setError('Choose a CSV file first.');
    setLoading(true);
    try {
      const body = new FormData(); body.append('file', file); body.append('question', question); body.append('research', research); if (sheetName.trim()) body.append('sheet_name', sheetName.trim());
      const response = await fetch(`${API_URL}/analyze`, { method: 'POST', body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Analysis failed.');
      setResult(data);
    } catch (err) { setError(err.message); } finally { setLoading(false); }
  }
  return <main>
    <header><span className="eyebrow">AI DATA ANALYST</span><h1>Turn your CSV into a clear business story.</h1><p>Upload data, ask a question, and get quality checks, metrics, trends, and practical next steps.</p></header>
    <section className="card form-card"><form onSubmit={submit}>
      <label className="upload"> <input type="file" accept=".csv,.tsv,.txt,.xlsx,.xls,.json,.ndjson,.parquet,text/csv,text/tab-separated-values,application/json,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.ms-excel" onChange={e => setFile(e.target.files?.[0])}/><strong>{file ? file.name : 'Choose a data file'}</strong><span>CSV, TSV, Excel, JSON, or Parquet</span></label>
      <label>Excel sheet <input value={sheetName} onChange={e => setSheetName(e.target.value)} placeholder="First sheet by default" /></label>
      <label>Question<textarea value={question} onChange={e => setQuestion(e.target.value)} rows="3" /></label>
      <label className="check"><input type="checkbox" checked={research} onChange={e => setResearch(e.target.checked)} /> Include external context when configured</label>
      <button disabled={loading}>{loading ? 'Analyzing data…' : 'Analyze dataset'}</button>
    </form></section>
    {error && <p className="error">{error}</p>}
    {result && <section className="results"><div className="card"><span className="eyebrow">EXECUTIVE SUMMARY</span><h2>{result.filename}</h2><p>{result.summary}</p></div>
      <div className="metrics">{result.metrics.map(item => <div className="metric" key={item.label}><span>{item.label}</span><strong>{item.value}</strong></div>)}</div>
      <div className="two-col"><article className="card"><h2>Key findings</h2><ul>{result.insights.map((x, i) => <li key={i}>{x}</li>)}</ul><h2>Recommendations</h2><ul>{result.recommendations.map((x, i) => <li key={i}>{x}</li>)}</ul></article>
      <article className="card"><h2>Data quality</h2><p><strong>{result.profile.rows.toLocaleString()}</strong> rows · <strong>{result.profile.columns}</strong> columns · <strong>{result.profile.duplicates}</strong> duplicate rows</p><div className="columns">{result.profile.column_profile.map(c => <div key={c.name}><strong>{c.name}</strong><span>{c.dtype} · {c.missing_pct}% missing</span></div>)}</div></article></div>
      {result.chart_url && <iframe className="chart" title="Analysis chart" src={`${API_URL}${result.chart_url}`} />}
      {result.research && <article className="card research"><h2>External context</h2><p>{result.research}</p></article>}
    </section>}
  </main>;
}
createRoot(document.getElementById('root')).render(<App />);
