import React from 'react';
import jsPDF from 'jspdf';
import ScoreGauge from './ScoreGauge';
import ExplanationPanel from './ExplanationPanel';
import HighlightedText from './HighlightedText';

// ─── Colour palette (matches dark UI) ──────────────────────────────
const C = {
    indigo: [99, 102, 241],
    indigoDark: [67, 56, 202],
    purple: [139, 92, 246],
    green: [34, 197, 94],
    amber: [245, 158, 11],
    red: [239, 68, 68],
    white: [255, 255, 255],
    offWhite: [248, 250, 252],
    slate100: [241, 245, 249],
    slate200: [226, 232, 240],
    slate400: [148, 163, 184],
    slate500: [100, 116, 139],
    slate600: [71, 85, 105],
    slate700: [51, 65, 85],
    slate800: [30, 41, 59],
    slate900: [15, 23, 42],
    bgDark: [10, 10, 20],
};

// Score-dependent colours
function scoreColor(score) {
    return score < 41 ? C.green : score < 71 ? C.amber : C.red;
}
function scoreLabel(score) {
    return score < 41 ? 'Likely Human' : score < 71 ? 'Uncertain / Mixed' : 'Likely AI-Generated';
}
function riskColor(risk) {
    return risk < 40 ? C.green : risk < 70 ? C.amber : C.red;
}
function riskLabel(risk) {
    return risk < 40 ? 'Low' : risk < 70 ? 'Medium' : 'High';
}

// ─── Helper: draw a filled rounded-rect ────────────────────────────
function roundRect(doc, x, y, w, h, r, fill) {
    doc.setFillColor(...fill);
    doc.roundedRect(x, y, w, h, r, r, 'F');
}

// ─── Helper: draw score arc (approximated with line segments) ──────
function drawArc(doc, cx, cy, r, from, to, color, lw) {
    doc.setDrawColor(...color);
    doc.setLineWidth(lw);
    const steps = 60;
    const pts = [];
    for (let i = 0; i <= steps; i++) {
        const a = from + (to - from) * (i / steps);
        pts.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
    }
    for (let i = 0; i < pts.length - 1; i++) {
        doc.line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]);
    }
}

// ─── Load image as data URL ────────────────────────────────────────
async function loadImageAsDataURL(src) {
    const res = await fetch(src);
    const blob = await res.blob();
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result);
        reader.readAsDataURL(blob);
    });
}

// ─── Main export function ───────────────────────────────────────────
async function generatePDF(result) {
    const { score, verdict, explanation, highlighted_text, gltr_score, processing_time_ms } = result;
    const doc = new jsPDF({ unit: 'pt', format: 'a4', orientation: 'portrait' });
    const pw = doc.internal.pageSize.getWidth();   // 595
    const ph = doc.internal.pageSize.getHeight();  // 842
    const ml = 42;   // left margin
    const mr = 42;   // right margin
    const uw = pw - ml - mr;  // usable width

    const now = new Date();
    const date = now.toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric' });
    const time = now.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
    const words = highlighted_text?.reduce((a, s) => a + s.sentence.split(' ').length, 0) ?? 0;
    const sents = highlighted_text?.length ?? 0;

    let page = 1;

    // ----------------------------------------------------------------
    // 1. COVER HEADER BAND
    // ----------------------------------------------------------------
    // top gradient strip
    doc.setFillColor(...C.indigoDark);
    doc.rect(0, 0, pw, 110, 'F');
    doc.setFillColor(...C.indigo);
    doc.rect(0, 80, pw, 30, 'F');
    // subtle diagonal accent
    doc.setFillColor(...C.purple);
    doc.triangle(pw - 160, 0, pw, 0, pw, 110, 'F');
    // load logo
    let logoDataUrl = null;
    try { logoDataUrl = await loadImageAsDataURL('/favicon.png'); } catch (_) { }

    // small logo in header
    if (logoDataUrl) {
        doc.addImage(logoDataUrl, 'PNG', ml, 22, 40, 40);
    } else {
        roundRect(doc, ml, 24, 38, 38, 6, [255, 255, 255, 0.15]);
        doc.setFillColor(255, 255, 255);
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(18);
        doc.setTextColor(255, 255, 255);
        doc.text('DV', ml + 19, 48, { align: 'center' });
    }
    // product name
    doc.setFontSize(22);
    doc.text('DeepVerify', ml + 48, 40);
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(9.5);
    doc.setTextColor(200, 210, 255);
    doc.text('AI Text Authenticity Platform  ·  Sprint 1 Demo  ·  SDG 16', ml + 48, 55);
    // report title
    doc.setFontSize(11);
    doc.setTextColor(255, 255, 255);
    doc.text('AUTHENTICITY ANALYSIS REPORT', ml + 48, 70);
    // date/time top-right
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(8);
    doc.setTextColor(200, 210, 255);
    doc.text(`${date}  ${time}`, pw - mr, 22, { align: 'right' });
    doc.text(`Page ${page}`, pw - mr, 32, { align: 'right' });

    let y = 128;

    // ----------------------------------------------------------------
    // 2. METADATA ROW  (4 chips)
    // ----------------------------------------------------------------
    const meta = [
        { label: 'Words Analysed', value: words.toLocaleString() },
        { label: 'Sentences', value: sents.toString() },
        { label: 'GLTR Score', value: gltr_score + ' / 100' },
        { label: 'Processing', value: (processing_time_ms ?? 0) + ' ms' },
    ];
    const chipW = (uw - 9) / 4;
    meta.forEach((m, i) => {
        const cx = ml + i * (chipW + 3);
        roundRect(doc, cx, y, chipW, 38, 5, C.slate100);
        doc.setFont('helvetica', 'bold'); doc.setFontSize(14); doc.setTextColor(...C.slate800);
        doc.text(m.value, cx + chipW / 2, y + 16, { align: 'center' });
        doc.setFont('helvetica', 'normal'); doc.setFontSize(7.5); doc.setTextColor(...C.slate500);
        doc.text(m.label.toUpperCase(), cx + chipW / 2, y + 28, { align: 'center' });
    });
    y += 52;

    // ----------------------------------------------------------------
    // 3. SCORE PANEL  (arc gauge + verdict block side by side)
    // ----------------------------------------------------------------
    const panelH = 145;
    const leftW = 200;
    const rightW = uw - leftW - 12;

    // left card – arc gauge
    roundRect(doc, ml, y, leftW, panelH, 8, C.offWhite);
    doc.setDrawColor(...C.slate200); doc.setLineWidth(0.5);
    doc.roundedRect(ml, y, leftW, panelH, 8, 8, 'S');

    const cx = ml + leftW / 2;
    const cy = y + 80;
    const rOuter = 48;
    const arcFrom = Math.PI * 0.75;   // 135°
    const arcTo = Math.PI * 2.25;   // 405° (≡ 45°)
    // background arc
    drawArc(doc, cx, cy, rOuter, arcFrom, arcTo, C.slate200, 8);
    // foreground arc (score fraction)
    const scoreFrac = Math.min(score / 100, 1);
    drawArc(doc, cx, cy, rOuter, arcFrom, arcFrom + (arcTo - arcFrom) * scoreFrac, scoreColor(score), 8);
    // centre score text
    doc.setFont('helvetica', 'bold'); doc.setFontSize(28); doc.setTextColor(...scoreColor(score));
    doc.text(String(score), cx, cy + 9, { align: 'center' });
    doc.setFont('helvetica', 'normal'); doc.setFontSize(7); doc.setTextColor(...C.slate500);
    doc.text('AI CONFIDENCE', cx, cy + 20, { align: 'center' });
    // threshold legend below arc
    const threshY = y + panelH - 18;
    [['0–40', C.green, ml + 24], ['41–70', C.amber, ml + 80], ['71–100', C.red, ml + 148]].forEach(([lbl, col, lx]) => {
        doc.setFillColor(...col); doc.circle(lx, threshY + 1, 3, 'F');
        doc.setFont('helvetica', 'normal'); doc.setFontSize(7); doc.setTextColor(...C.slate500);
        doc.text(lbl, lx + 5, threshY + 3);
    });

    // right card – verdict details
    const rx = ml + leftW + 12;
    roundRect(doc, rx, y, rightW, panelH, 8, C.offWhite);
    doc.setDrawColor(...C.slate200); doc.roundedRect(rx, y, rightW, panelH, 8, 8, 'S');

    // verdict badge
    const sc = scoreColor(score);
    roundRect(doc, rx + 14, y + 14, rightW - 28, 26, 5, [...sc.map(v => v), 0.12]);
    doc.setFont('helvetica', 'bold'); doc.setFontSize(13); doc.setTextColor(...sc);
    doc.text(scoreLabel(score).toUpperCase(), rx + rightW / 2, y + 31.5, { align: 'center' });

    // metrics grid
    const mItems = [
        { lbl: 'Detection Engine', val: 'GLTR + Heuristics' },
        { lbl: 'Model', val: 'GPT-2 (Sprint 1)' },
        { lbl: 'Composite Score', val: `${score} / 100` },
        { lbl: 'GLTR Raw', val: `${gltr_score} / 100` },
        { lbl: 'Verdict', val: scoreLabel(score) },
        { lbl: 'Sprint', val: '1.0 (Demo)' },
    ];
    const cols = 2;
    const itemW = (rightW - 28) / cols;
    mItems.forEach((m, i) => {
        const col = i % cols;
        const row = Math.floor(i / cols);
        const ix = rx + 14 + col * itemW;
        const iy = y + 52 + row * 28;
        doc.setFont('helvetica', 'normal'); doc.setFontSize(7.5); doc.setTextColor(...C.slate500);
        doc.text(m.lbl.toUpperCase(), ix, iy);
        doc.setFont('helvetica', 'bold'); doc.setFontSize(9); doc.setTextColor(...C.slate800);
        doc.text(m.val, ix, iy + 12);
    });
    y += panelH + 16;

    // ----------------------------------------------------------------
    // 4. SCORE BAR VISUAL
    // ----------------------------------------------------------------
    doc.setFont('helvetica', 'bold'); doc.setFontSize(9); doc.setTextColor(...C.slate700);
    doc.text('AI PROBABILITY SCALE', ml, y);
    y += 8;
    const barH = 14;
    // gradient background bar (approximated with coloured segments)
    const segments = [
        { from: 0, to: 0.40, color: [...C.green] },
        { from: 0.40, to: 0.70, color: [...C.amber] },
        { from: 0.70, to: 1.0, color: [...C.red] },
    ];
    segments.forEach(s => {
        doc.setFillColor(...s.color, 40);
        doc.rect(ml + s.from * uw, y, (s.to - s.from) * uw, barH, 'F');
    });
    // outer border
    doc.setDrawColor(...C.slate200); doc.setLineWidth(0.5);
    doc.roundedRect(ml, y, uw, barH, 2, 2, 'S');
    // score marker
    const markerX = ml + (score / 100) * uw;
    doc.setFillColor(...scoreColor(score));
    doc.triangle(markerX - 5, y - 5, markerX + 5, y - 5, markerX, y, 'F');
    roundRect(doc, markerX - 12, y - 18, 24, 12, 3, scoreColor(score));
    doc.setFont('helvetica', 'bold'); doc.setFontSize(7.5); doc.setTextColor(255, 255, 255);
    doc.text(String(score), markerX, y - 9, { align: 'center' });
    // labels
    doc.setFont('helvetica', 'normal'); doc.setFontSize(7); doc.setTextColor(...C.green);
    doc.text('Human', ml + 2, y + barH + 10);
    doc.setTextColor(...C.amber);
    doc.text('Uncertain', ml + 0.40 * uw + 2, y + barH + 10);
    doc.setTextColor(...C.red);
    doc.text('AI', ml + 0.70 * uw + 2, y + barH + 10);
    y += barH + 20;

    // ----------------------------------------------------------------
    // 5. AI EXPLANATION SECTION
    // ----------------------------------------------------------------
    roundRect(doc, ml, y, uw, 12, 0, C.indigoDark);
    doc.setFont('helvetica', 'bold'); doc.setFontSize(8.5); doc.setTextColor(255, 255, 255);
    doc.text('  AI EXPLANATION', ml + 4, y + 8.5);
    y += 16;
    if (explanation) {
        const cleanExp = explanation.replace(/\(Note:.*?\)/gi, '').trim();
        const lines = doc.setFont('helvetica', 'normal').setFontSize(9).splitTextToSize(cleanExp, uw - 16);
        const expH = lines.length * 13 + 16;
        roundRect(doc, ml, y, uw, expH, 4, C.offWhite);
        doc.setTextColor(...C.slate700);
        doc.text(lines, ml + 8, y + 12);
        y += expH + 10;
    }

    // ----------------------------------------------------------------
    // 6. SENTENCE-LEVEL ANALYSIS TABLE
    // ----------------------------------------------------------------
    // New page if needed
    if (y > ph - 200 && highlighted_text?.length) {
        doc.addPage(); page++;
        // small header repeat
        doc.setFillColor(...C.indigoDark); doc.rect(0, 0, pw, 24, 'F');
        doc.setFont('helvetica', 'bold'); doc.setFontSize(8); doc.setTextColor(255, 255, 255);
        doc.text('DeepVerify  ·  AI Authenticity Report (continued)', ml, 16);
        doc.text(`Page ${page}`, pw - mr, 16, { align: 'right' });
        y = 36;
    }

    roundRect(doc, ml, y, uw, 12, 0, C.indigoDark);
    doc.setFont('helvetica', 'bold'); doc.setFontSize(8.5); doc.setTextColor(255, 255, 255);
    doc.text('  SENTENCE-LEVEL ANALYSIS', ml + 4, y + 8.5);
    y += 16;

    if (highlighted_text?.length) {
        // Column headers
        const colWidths = [32, 62, uw - 94];  // Risk %, Label, Sentence
        const colX = [ml, ml + 32, ml + 94];
        roundRect(doc, ml, y, uw, 16, 0, C.slate800);
        const colHeaders = ['RISK %', 'RISK LEVEL', 'SENTENCE'];
        colHeaders.forEach((h, i) => {
            doc.setFont('helvetica', 'bold'); doc.setFontSize(7.5); doc.setTextColor(...C.slate200);
            doc.text(h, colX[i] + 3, y + 10.5);
        });
        y += 16;

        highlighted_text.forEach((item, idx) => {
            const risk = item.risk;
            const color = riskColor(risk);
            const level = riskLabel(risk);
            const sentLines = doc.setFont('helvetica', 'normal').setFontSize(8.5)
                .splitTextToSize(item.sentence, colWidths[2] - 6);
            const rowH = Math.max(sentLines.length * 12 + 8, 22);

            // Check for page break
            if (y + rowH > ph - 50) {
                doc.addPage(); page++;
                doc.setFillColor(...C.indigoDark); doc.rect(0, 0, pw, 24, 'F');
                doc.setFont('helvetica', 'bold'); doc.setFontSize(8); doc.setTextColor(255, 255, 255);
                doc.text('DeepVerify  ·  Sentence Analysis (continued)', ml, 16);
                doc.text(`Page ${page}`, pw - mr, 16, { align: 'right' });
                y = 36;
                // Repeat header
                roundRect(doc, ml, y, uw, 16, 0, C.slate800);
                colHeaders.forEach((h, i) => {
                    doc.setFont('helvetica', 'bold'); doc.setFontSize(7.5); doc.setTextColor(...C.slate200);
                    doc.text(h, colX[i] + 3, y + 10.5);
                });
                y += 16;
            }

            // Row background
            const rowBg = idx % 2 === 0 ? C.offWhite : C.white;
            roundRect(doc, ml, y, uw, rowH, 0, rowBg);
            // left risk colour strip
            doc.setFillColor(...color);
            doc.rect(ml, y, 3, rowH, 'F');

            // Risk % badge
            roundRect(doc, colX[0] + 4, y + (rowH - 14) / 2, 24, 14, 3, [...color.map(v => Math.round(v * 0.2 + 200))]);
            doc.setFont('helvetica', 'bold'); doc.setFontSize(8); doc.setTextColor(...color);
            doc.text(`${risk}%`, colX[0] + 16, y + (rowH - 14) / 2 + 9, { align: 'center' });

            // Risk label
            doc.setFont('helvetica', 'normal'); doc.setFontSize(8); doc.setTextColor(...color);
            doc.text(level, colX[1] + 3, y + rowH / 2 + 3);

            // Sentence text
            doc.setFont('helvetica', 'normal'); doc.setFontSize(8.5); doc.setTextColor(...C.slate700);
            doc.text(sentLines, colX[2] + 3, y + 12);

            // Row divider
            doc.setDrawColor(...C.slate200); doc.setLineWidth(0.3);
            doc.line(ml, y + rowH, ml + uw, y + rowH);

            y += rowH;
        });
    }

    y += 16;

    // ----------------------------------------------------------------
    // 7. RISK DISTRIBUTION SUMMARY (bar chart)
    // ----------------------------------------------------------------
    if (highlighted_text?.length) {
        if (y > ph - 120) { doc.addPage(); page++; y = 40; }
        const low = highlighted_text.filter(s => s.risk < 40).length;
        const medium = highlighted_text.filter(s => s.risk >= 40 && s.risk < 70).length;
        const high = highlighted_text.filter(s => s.risk >= 70).length;
        const total = highlighted_text.length;

        roundRect(doc, ml, y, uw, 12, 0, C.indigoDark);
        doc.setFont('helvetica', 'bold'); doc.setFontSize(8.5); doc.setTextColor(255, 255, 255);
        doc.text('  RISK DISTRIBUTION SUMMARY', ml + 4, y + 8.5);
        y += 16;
        roundRect(doc, ml, y, uw, 56, 4, C.offWhite);

        const barItems = [
            { label: 'Low Risk (Human)', count: low, color: C.green },
            { label: 'Medium Risk', count: medium, color: C.amber },
            { label: 'High Risk (AI)', count: high, color: C.red },
        ];
        const bw = (uw - 32) / 3;
        barItems.forEach((item, i) => {
            const bx = ml + 8 + i * (bw + 8);
            const pct = total > 0 ? (item.count / total * 100).toFixed(0) : '0';
            roundRect(doc, bx, y + 8, bw, 26, 4, [...item.color]);
            doc.setFont('helvetica', 'bold'); doc.setFontSize(14); doc.setTextColor(255, 255, 255);
            doc.text(`${item.count}`, bx + bw / 2, y + 24, { align: 'center' });
            doc.setFont('helvetica', 'normal'); doc.setFontSize(7.5); doc.setTextColor(...C.slate600);
            doc.text(`${item.label} (${pct}%)`, bx + bw / 2, y + 46, { align: 'center' });
        });
        y += 70;
    }

    // ----------------------------------------------------------------
    // 8. FOOTER on each page (last page)
    // ----------------------------------------------------------------
    const totalPages = doc.internal.getNumberOfPages();
    for (let p = 1; p <= totalPages; p++) {
        doc.setPage(p);
        const fy = ph - 28;
        doc.setFillColor(...C.slate100);
        doc.rect(0, fy - 4, pw, 32, 'F');
        doc.setDrawColor(...C.slate200); doc.setLineWidth(0.4);
        doc.line(0, fy - 4, pw, fy - 4);
        doc.setFont('helvetica', 'bold'); doc.setFontSize(7.5); doc.setTextColor(...C.indigo);
        doc.text('DeepVerify', ml, fy + 6);
        doc.setFont('helvetica', 'normal'); doc.setTextColor(...C.slate500);
        doc.text(`AI Authenticity Report  ·  ${date}`, ml + 60, fy + 6);
        doc.text('Aligned with SDG 16: Peace, Justice & Strong Institutions', ml, fy + 17);
        doc.text(`Page ${p} of ${totalPages}`, pw - mr, fy + 6, { align: 'right' });
        doc.setFontSize(6.5); doc.setTextColor(...C.slate400);
        doc.text('This report is generated by an automated AI detection system. Results are indicative and should not be used as sole evidence of authorship.', ml, fy + 17, { maxWidth: uw - 60 });
    }

    const ts = now.toISOString().slice(0, 16).replace('T', '_').replace(':', '');
    doc.save(`deepverify-report-${ts}.pdf`);
}

// ─── Component ─────────────────────────────────────────────────────
export default function ResultsPanel({ result }) {
    if (!result) return null;
    const { score, verdict, explanation, highlighted_text, gltr_score } = result;

    return (
        <div className="fade-in-up" style={{ display: 'flex', flexDirection: 'column', gap: 'clamp(12px,3vw,20px)' }}>
            {/* Header */}
            <div className="flex items-center justify-between flex-wrap gap-2">
                <h2 style={{ fontSize: 'var(--text-xl)', fontWeight: 600 }}>Analysis Results</h2>
                <button
                    id="export-pdf-btn"
                    onClick={() => generatePDF(result).catch(console.error)}
                    className="flex items-center gap-2 rounded-xl font-medium transition-all"
                    style={{ padding: 'clamp(8px,1.5vw,10px) clamp(12px,2.5vw,18px)', fontSize: 'var(--text-sm)', background: 'rgba(99,102,241,0.1)', color: 'var(--accent-blue-light)', border: '1px solid rgba(99,102,241,0.25)', minHeight: '40px' }}
                    aria-label="Download analysis as PDF"
                >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                        <polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" />
                    </svg>
                    Export PDF Report
                </button>
            </div>

            {/* Score + Explanation */}
            <div className="results-grid">
                <ScoreGauge score={score} verdict={verdict} />
                <ExplanationPanel explanation={explanation} score={score} gltrScore={gltr_score} />
            </div>

            <HighlightedText highlightedText={highlighted_text} />
        </div>
    );
}
