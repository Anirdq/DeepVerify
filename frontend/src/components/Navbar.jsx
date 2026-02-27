import React from 'react';

export default function Navbar({ theme, onToggleTheme }) {
    const isDark = theme === 'dark';

    return (
        <nav className="navbar">
            <div
                className="page-container flex items-center justify-between"
                style={{ height: 'clamp(56px, 8vw, 68px)', gap: '12px' }}
            >
                {/* ── Logo + Wordmark ── */}
                <a
                    href="/"
                    className="flex items-center flex-shrink-0"
                    style={{ gap: 'clamp(10px, 2vw, 14px)', textDecoration: 'none' }}
                    aria-label="DeepVerify home"
                >
                    {/* Logo image with glow ring */}
                    <div style={{ position: 'relative', flexShrink: 0 }}>
                        <div style={{
                            position: 'absolute', inset: -3,
                            borderRadius: 14,
                            background: 'linear-gradient(135deg, rgba(0,188,212,0.35), rgba(92,53,204,0.35))',
                            filter: 'blur(6px)',
                            zIndex: 0,
                        }} />
                        <div style={{
                            position: 'relative', zIndex: 1,
                            width: 42, height: 42, borderRadius: 12,
                            overflow: 'hidden',
                            border: '1.5px solid rgba(0,188,212,0.3)',
                            background: isDark ? 'rgba(10,10,20,0.6)' : 'rgba(248,250,252,0.8)',
                        }}>
                            <img
                                src="/favicon.png"
                                alt="DeepVerify"
                                width="42"
                                height="42"
                                style={{ display: 'block', width: '100%', height: '100%', objectFit: 'contain', padding: '3px' }}
                            />
                        </div>
                    </div>

                    {/* Text block */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1px' }}>
                        <div style={{
                            fontSize: 'clamp(1rem, 3.5vw, 1.2rem)',
                            fontWeight: 800,
                            letterSpacing: '-0.02em',
                            lineHeight: 1,
                            background: 'linear-gradient(135deg, #00bcd4 0%, #6366f1 55%, #a855f7 100%)',
                            WebkitBackgroundClip: 'text',
                            WebkitTextFillColor: 'transparent',
                            backgroundClip: 'text',
                        }}>
                            DeepVerify
                        </div>
                        <div style={{
                            fontSize: 'clamp(0.55rem, 1.5vw, 0.67rem)',
                            fontWeight: 500,
                            letterSpacing: '0.12em',
                            textTransform: 'uppercase',
                            color: 'var(--text-muted)',
                        }}>
                            AI Authenticity
                        </div>
                    </div>
                </a>

                {/* ── Right: theme toggle ── */}
                <button
                    id="theme-toggle-btn"
                    className="theme-toggle"
                    onClick={onToggleTheme}
                    aria-label={isDark ? 'Switch to light theme' : 'Switch to dark theme'}
                    title={isDark ? 'Light mode' : 'Dark mode'}
                >
                    {isDark ? (
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                            <circle cx="12" cy="12" r="5" />
                            <line x1="12" y1="1" x2="12" y2="3" />
                            <line x1="12" y1="21" x2="12" y2="23" />
                            <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
                            <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
                            <line x1="1" y1="12" x2="3" y2="12" />
                            <line x1="21" y1="12" x2="23" y2="12" />
                            <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
                            <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
                        </svg>
                    ) : (
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                            <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
                        </svg>
                    )}
                </button>
            </div>
        </nav>
    );
}
