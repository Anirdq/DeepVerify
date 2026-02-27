import React from 'react';

export default function ScoreGauge({ score }) {
    const radius = 52;
    const circumference = 2 * Math.PI * radius;
    const dashOffset = circumference * (1 - Math.min(Math.max(score / 100, 0), 1));

    const gaugeColor = score < 41 ? '#22c55e' : score < 71 ? '#f59e0b' : '#ef4444';
    const glowColor = score < 41 ? 'rgba(34,197,94,0.4)' : score < 71 ? 'rgba(245,158,11,0.4)' : 'rgba(239,68,68,0.4)';
    const verdictLabel = score < 41 ? 'Likely Human' : score < 71 ? 'Uncertain' : 'Likely AI';
    const verdictIcon = score < 41 ? '✓' : score < 71 ? '◈' : '⚠';

    const verdictStyle =
        score < 41
            ? { background: 'rgba(34,197,94,0.1)', color: '#22c55e', border: '1px solid rgba(34,197,94,0.25)' }
            : score < 71
                ? { background: 'rgba(245,158,11,0.1)', color: '#f59e0b', border: '1px solid rgba(245,158,11,0.25)' }
                : { background: 'rgba(239,68,68,0.1)', color: '#ef4444', border: '1px solid rgba(239,68,68,0.25)' };

    return (
        <div className="glass-card fade-in-up flex flex-col items-center" style={{ padding: 'clamp(14px,3vw,24px)', gap: 'clamp(10px,2vw,16px)' }}>
            <h3 style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--text-secondary)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
                AI Confidence Score
            </h3>

            {/* SVG gauge — scales with container */}
            <div className="relative flex items-center justify-center w-full" style={{ maxWidth: '160px' }}>
                <svg
                    viewBox="0 0 140 140"
                    style={{ width: '100%', height: 'auto', filter: `drop-shadow(0 0 14px ${glowColor})` }}
                    aria-label={`AI detection score: ${score} out of 100`}
                    role="img"
                >
                    <circle cx="70" cy="70" r={radius} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="10" />
                    <circle
                        cx="70" cy="70" r={radius}
                        fill="none" stroke={gaugeColor} strokeWidth="10"
                        strokeLinecap="round"
                        strokeDasharray={circumference}
                        strokeDashoffset={dashOffset}
                        className="score-ring-progress"
                        transform="rotate(-90 70 70)"
                    />
                </svg>
                <div className="absolute flex flex-col items-center">
                    <span style={{ fontSize: 'clamp(1.6rem,5vw,2.4rem)', fontWeight: 700, color: gaugeColor, lineHeight: 1 }}>{score}</span>
                    <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>/ 100</span>
                </div>
            </div>

            <div className="badge" style={verdictStyle}>
                {verdictIcon} {verdictLabel}
            </div>

            {/* Legend */}
            <div className="w-full grid grid-cols-3 gap-1" style={{ fontSize: 'var(--text-xs)', textAlign: 'center' }}>
                <div className="py-1 rounded" style={{ background: 'rgba(34,197,94,0.08)', color: '#4ade80' }}>0–40 Human</div>
                <div className="py-1 rounded" style={{ background: 'rgba(245,158,11,0.08)', color: '#fbbf24' }}>41–70 Mixed</div>
                <div className="py-1 rounded" style={{ background: 'rgba(239,68,68,0.08)', color: '#f87171' }}>71–100 AI</div>
            </div>
        </div>
    );
}
