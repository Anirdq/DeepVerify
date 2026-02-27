import React from 'react';

const getRiskClass = (risk) =>
    risk < 40 ? 'sentence-low' : risk < 70 ? 'sentence-medium' : 'sentence-high';

const getRiskLabel = (risk) =>
    risk < 40 ? `Low AI risk (${risk}%)` : risk < 70 ? `Medium AI risk (${risk}%)` : `High AI risk (${risk}%)`;

export default function HighlightedText({ highlightedText }) {
    if (!highlightedText?.length) return null;

    return (
        <div className="glass-card fade-in-up" style={{ padding: 'clamp(14px,3vw,24px)', animationDelay: '0.15s' }}>
            {/* Header + legend */}
            <div className="flex items-start sm:items-center justify-between flex-wrap gap-2 mb-4">
                <h3 style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--text-secondary)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
                    Sentence-Level Analysis
                </h3>
                <div className="flex items-center gap-3" style={{ fontSize: 'var(--text-xs)' }}>
                    {[
                        { label: 'Human', color: '#22c55e', bg: 'rgba(34,197,94,0.3)' },
                        { label: 'Uncertain', color: '#f59e0b', bg: 'rgba(245,158,11,0.3)' },
                        { label: 'AI', color: '#ef4444', bg: 'rgba(239,68,68,0.3)' },
                    ].map(({ label, color, bg }) => (
                        <span key={label} className="flex items-center gap-1" style={{ color }}>
                            <span className="inline-block rounded-sm" style={{ width: 10, height: 10, background: bg, flexShrink: 0 }} />
                            {label}
                        </span>
                    ))}
                </div>
            </div>

            <p style={{ fontSize: 'var(--text-base)', lineHeight: 2 }}>
                {highlightedText.map((item, i) => (
                    <span
                        key={i}
                        className={`tooltip cursor-pointer ${getRiskClass(item.risk)}`}
                        aria-label={getRiskLabel(item.risk)}
                    >
                        {item.sentence}{' '}
                        <span className="tooltip-content">{getRiskLabel(item.risk)}</span>
                    </span>
                ))}
            </p>
        </div>
    );
}
