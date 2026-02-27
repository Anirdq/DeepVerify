import React from 'react';

export default function ExplanationPanel({ explanation, score, gltrScore }) {
    if (!explanation) return null;

    return (
        <div className="glass-card fade-in-up" style={{ padding: 'clamp(14px,3vw,24px)', animationDelay: '0.1s', display: 'flex', flexDirection: 'column', gap: 'clamp(10px,2vw,16px)' }}>
            {/* Header */}
            <div className="flex items-center gap-2 flex-wrap">
                <div className="flex-shrink-0 flex items-center justify-center rounded-lg w-7 h-7"
                    style={{ background: 'linear-gradient(135deg,#8b5cf6,#c084fc)', boxShadow: '0 0 10px rgba(139,92,246,0.4)' }}>
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2">
                        <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
                        <line x1="12" y1="17" x2="12.01" y2="17" />
                        <circle cx="12" cy="12" r="10" />
                    </svg>
                </div>
                <h3 style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--text-secondary)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
                    AI Explanation
                </h3>
                <div className="ml-auto badge" style={{ background: 'rgba(139,92,246,0.1)', color: '#c084fc', border: '1px solid rgba(139,92,246,0.2)', fontSize: '0.62rem' }}>
                    Mistral 7B
                </div>
            </div>

            <p style={{ fontSize: 'var(--text-sm)', lineHeight: 1.75, color: 'var(--text-primary)' }}>
                {explanation}
            </p>

            {/* Score breakdown */}
            {gltrScore !== undefined && (
                <div style={{ paddingTop: 'clamp(8px,1.5vw,12px)', borderTop: '1px solid var(--border-subtle)' }}>
                    <p style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', marginBottom: '8px' }}>DETECTION METHOD BREAKDOWN</p>
                    <div className="grid grid-cols-2 gap-2 sm:gap-3">
                        {[
                            { label: 'GLTR Score', value: gltrScore, sub: 'Token Probability' },
                            { label: 'Composite', value: score, sub: 'Final AI Score' },
                        ].map((item) => (
                            <div key={item.label} className="rounded-lg" style={{ padding: 'clamp(8px,1.5vw,12px)', background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border-subtle)' }}>
                                <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', marginBottom: '4px' }}>{item.label}</div>
                                <div style={{ fontSize: 'clamp(1.1rem,3.5vw,1.4rem)', fontWeight: 700, color: 'var(--accent-blue-light)' }}>{item.value}</div>
                                <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>{item.sub}</div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
