import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import TextInput from './components/TextInput';
import ResultsPanel from './components/ResultsPanel';

export default function App() {
  // ── Theme state: read from localStorage or default to 'dark'
  const [theme, setTheme] = useState(() => localStorage.getItem('dv-theme') || 'dark');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Apply theme to <html> element
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('dv-theme', theme);
  }, [theme]);

  const toggleTheme = () => setTheme(t => t === 'dark' ? 'light' : 'dark');

  const handleAnalyse = async (text) => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${response.status}`);
      }
      setResult(await response.json());
    } catch (err) {
      setError(err.message || 'Failed to analyse text. Please ensure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen animated-bg flex flex-col">
      <Navbar theme={theme} onToggleTheme={toggleTheme} />

      <main
        className="flex-1 page-container flex flex-col"
        style={{ gap: 'clamp(20px,4vw,32px)', paddingTop: 'clamp(32px,6vw,56px)', paddingBottom: 'clamp(24px,5vw,48px)' }}
      >
        {/* ── Hero ── */}
        <div className="text-center flex flex-col items-center" style={{ gap: 'clamp(8px,2vw,12px)' }}>
          <h1 style={{ fontSize: 'var(--text-hero)', fontWeight: 800, lineHeight: 1.15 }}>
            Is this text{' '}
            <span className="gradient-text">AI-generated?</span>
          </h1>
          <p className="mx-auto" style={{ fontSize: 'var(--text-base)', color: 'var(--text-secondary)', maxWidth: '520px', lineHeight: 1.65 }}>
            Paste any text and DeepVerify will analyse it using statistical token-probability methods,
            giving you an <strong style={{ color: 'var(--text-primary)' }}>AI-authenticity score</strong> in seconds.
          </p>
        </div>

        {/* ── Input ── */}
        <TextInput onAnalyse={handleAnalyse} loading={loading} />

        {/* ── Loading skeleton ── */}
        {loading && (
          <div className="glass-card" style={{ padding: 'clamp(16px,3vw,24px)', display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div className="flex items-center gap-3">
              <div className="w-7 h-7 rounded-full shimmer flex-shrink-0" />
              <div className="flex-1 h-4 rounded shimmer" />
            </div>
            <div className="results-grid">
              <div className="shimmer rounded-xl" style={{ height: 'clamp(120px,20vw,150px)' }} />
              <div className="shimmer rounded-xl" style={{ height: 'clamp(120px,20vw,150px)' }} />
            </div>
            <div className="shimmer rounded-xl" style={{ height: '80px' }} />
            <p className="text-center" style={{ fontSize: 'var(--text-sm)', color: 'var(--text-muted)' }}>
              Running GLTR analysis…
            </p>
          </div>
        )}

        {/* ── Error ── */}
        {error && !loading && (
          <div
            className="glass-card fade-in-up"
            role="alert"
            style={{ padding: 'clamp(14px,3vw,20px)', display: 'flex', alignItems: 'flex-start', gap: '12px', border: '1px solid rgba(239,68,68,0.3)' }}
          >
            <div className="flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'rgba(239,68,68,0.1)' }}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#ef4444" strokeWidth="2">
                <circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
            </div>
            <div>
              <p className="font-semibold" style={{ fontSize: 'var(--text-sm)', color: '#f87171' }}>Analysis Failed</p>
              <p style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)', marginTop: '4px' }}>{error}</p>
              <p style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', marginTop: '8px' }}>
                Make sure the FastAPI backend is running at <code>http://localhost:8000</code>
              </p>
            </div>
          </div>
        )}

        {/* ── Results ── */}
        {result && !loading && <ResultsPanel result={result} />}

        {/* ── Footer ── */}
        <p className="text-center" style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
          DeepVerify promotes digital trust —{' '}
          <span style={{ color: 'var(--accent-blue-light)' }}>SDG 16: Peace, Justice &amp; Strong Institutions</span>
        </p>
      </main>
    </div>
  );
}
