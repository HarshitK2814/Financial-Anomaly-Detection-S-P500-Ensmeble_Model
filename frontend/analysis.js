const API_BASE = 'http://localhost:8000/api/ablation';

let ablationChart = null;
let expertChart = null;
let importanceChart = null;

const els = {
    dataset: document.getElementById('a-dataset'),
    qThresh: document.getElementById('a-q-threshold'),
    qVal: document.getElementById('a-q-value'),
    volLow: document.getElementById('a-vol-low'),
    volHigh: document.getElementById('a-vol-high'),
    srcLive: document.getElementById('src-live'),
    srcJournal: document.getElementById('src-journal'),
    journalNotice: document.getElementById('journal-notice')
};

let useJournal = false;

document.addEventListener('DOMContentLoaded', () => {
    fetchAblation();
    els.dataset.addEventListener('change', fetchAblation);
    els.qThresh.addEventListener('input', (e) => { els.qVal.textContent = e.target.value; });
    els.qThresh.addEventListener('change', fetchAblation);
    els.volLow.addEventListener('change', fetchAblation);
    els.volHigh.addEventListener('change', fetchAblation);

    els.srcLive.addEventListener('click', () => setSource(false));
    els.srcJournal.addEventListener('click', () => setSource(true));

    function setSource(isJournal) {
        useJournal = isJournal;
        if (useJournal) {
            els.srcJournal.classList.add('active');
            els.srcJournal.style.background = 'var(--accent)';
            els.srcJournal.style.color = 'white';
            els.srcLive.classList.remove('active');
            els.srcLive.style.background = 'transparent';
            els.srcLive.style.color = 'var(--text-muted)';
            els.journalNotice.style.display = 'block';
            // Dataset selector stays ENABLED - each window has different paper metrics
            els.qThresh.disabled = true;
            els.volLow.disabled = true;
            els.volHigh.disabled = true;
        } else {
            els.srcLive.classList.add('active');
            els.srcLive.style.background = 'var(--accent)';
            els.srcLive.style.color = 'white';
            els.srcJournal.classList.remove('active');
            els.srcJournal.style.background = 'transparent';
            els.srcJournal.style.color = 'var(--text-muted)';
            els.journalNotice.style.display = 'none';
            els.qThresh.disabled = false;
            els.volLow.disabled = false;
            els.volHigh.disabled = false;
        }
        fetchAblation();
    }
});

async function fetchAblation() {
    const params = new URLSearchParams({
        dataset: els.dataset.value,
        q_threshold: els.qThresh.value,
        vol_q_low: els.volLow.value,
        vol_q_high: els.volHigh.value,
        use_journal: useJournal
    });

    try {
        const res = await fetch(`${API_BASE}?${params.toString()}`);
        const json = await res.json();
        
        if (json.status === 'success') {
            console.log('Ablation data received:', json.data);
            renderAll(json.data);
        } else {
            console.error('API Error:', json.message);
            alert('Error loading analysis data: ' + json.message);
        }
    } catch (err) {
        console.error('Fetch error:', err);
        alert('Failed to fetch analysis data. Make sure backend is running on port 8000.');
    }
}

function pct(v) { return (v * 100).toFixed(1) + '%'; }

function renderAll(data) {
    const moe = data.moe;
    const single = data.single;

    // ============ Section 1: Ablation Metrics ============
    document.getElementById('moe-precision').textContent = pct(moe.overall.precision);
    document.getElementById('moe-recall').textContent = pct(moe.overall.recall);
    document.getElementById('moe-f1').textContent = pct(moe.overall.f1);
    
    document.getElementById('single-precision').textContent = pct(single.overall.precision);
    document.getElementById('single-recall').textContent = pct(single.overall.recall);
    document.getElementById('single-f1').textContent = pct(single.overall.f1);

    // Color the F1 scores
    const moeF1El = document.getElementById('moe-f1');
    const singleF1El = document.getElementById('single-f1');
    moeF1El.style.color = moe.overall.f1 > single.overall.f1 ? 'var(--success)' : 'var(--danger)';
    singleF1El.style.color = single.overall.f1 > moe.overall.f1 ? 'var(--success)' : 'var(--danger)';

    // Summary card
    const card = document.getElementById('summary-card');
    const text = document.getElementById('summary-text');
    card.style.display = 'block';
    const f1Diff = ((moe.overall.f1 - single.overall.f1) * 100).toFixed(1);
    if (parseFloat(f1Diff) > 0) {
        text.innerHTML = `The Regime-Conditioned MoE outperforms the single model by <strong style="color:var(--success)">+${f1Diff}pp</strong> in F1 Score, validating the hypothesis that regime-specific experts capture market dynamics that a monolithic model cannot.`;
    } else {
        text.innerHTML = `The models perform comparably on this configuration (Δ = ${f1Diff}pp). Try adjusting the regime separation quantiles to create more distinct volatility regimes.`;
    }

    // Ablation bar chart
    renderAblationChart(moe.overall, single.overall);

    // ============ Section 2: Per-Expert ============
    const experts = moe.per_expert;
    for (const [regime, el_prefix] of [['Low', 'low'], ['Medium', 'med'], ['High', 'high']]) {
        const e = experts[regime] || { precision: 0, recall: 0, f1: 0, n_samples: 0 };
        document.getElementById(`exp-${el_prefix}-prec`).textContent = pct(e.precision);
        document.getElementById(`exp-${el_prefix}-rec`).textContent = pct(e.recall);
        document.getElementById(`exp-${el_prefix}-f1`).textContent = pct(e.f1);
        document.getElementById(`exp-${el_prefix}-n`).textContent = e.n_samples;
    }

    renderExpertChart(experts);

    // ============ Section 3: Feature Importance ============
    renderImportanceChart(moe.importances, single.importances);
}

function renderAblationChart(moe, single) {
    const ctx = document.getElementById('ablationChart').getContext('2d');
    if (ablationChart) ablationChart.destroy();

    ablationChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Precision', 'Recall', 'F1 Score'],
            datasets: [
                {
                    label: 'MoE (Ours)',
                    data: [moe.precision, moe.recall, moe.f1],
                    backgroundColor: ['rgba(59, 130, 246, 0.7)', 'rgba(59, 130, 246, 0.7)', 'rgba(16, 185, 129, 0.7)'],
                    borderColor: ['rgba(59, 130, 246, 1)', 'rgba(59, 130, 246, 1)', 'rgba(16, 185, 129, 1)'],
                    borderWidth: 1,
                    borderRadius: 6,
                    barPercentage: 0.6
                },
                {
                    label: 'Single RF (Baseline)',
                    data: [single.precision, single.recall, single.f1],
                    backgroundColor: ['rgba(239, 68, 68, 0.4)', 'rgba(239, 68, 68, 0.4)', 'rgba(239, 68, 68, 0.5)'],
                    borderColor: ['rgba(239, 68, 68, 0.8)', 'rgba(239, 68, 68, 0.8)', 'rgba(239, 68, 68, 0.8)'],
                    borderWidth: 1,
                    borderRadius: 6,
                    barPercentage: 0.6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#94a3b8' } },
                title: { display: true, text: 'Ablation: MoE vs Monolithic Model Performance', color: '#e2e8f0', font: { size: 14 } },
                tooltip: {
                    backgroundColor: 'rgba(15, 17, 21, 0.9)',
                    callbacks: { label: (c) => `${c.dataset.label}: ${(c.raw * 100).toFixed(1)}%` }
                }
            },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                y: { 
                    grid: { color: 'rgba(255,255,255,0.05)' }, 
                    min: 0, max: 1.0, 
                    ticks: { 
                        color: '#94a3b8', 
                        callback: (v) => (v * 100).toFixed(0) + '%',
                        stepSize: 0.2
                    }
                }
            }
        }
    });
}

function renderExpertChart(experts) {
    const ctx = document.getElementById('expertChart').getContext('2d');
    if (expertChart) expertChart.destroy();

    const labels = ['Low Vol (E₀)', 'Medium Vol (E₁)', 'High Vol (E₂)'];
    const prec = [experts.Low?.precision || 0, experts.Medium?.precision || 0, experts.High?.precision || 0];
    const rec = [experts.Low?.recall || 0, experts.Medium?.recall || 0, experts.High?.recall || 0];
    const f1 = [experts.Low?.f1 || 0, experts.Medium?.f1 || 0, experts.High?.f1 || 0];

    expertChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Precision', data: prec,
                    backgroundColor: 'rgba(59, 130, 246, 0.6)', borderColor: 'rgba(59, 130, 246, 1)', borderWidth: 1, borderRadius: 4
                },
                {
                    label: 'Recall', data: rec,
                    backgroundColor: 'rgba(245, 158, 11, 0.6)', borderColor: 'rgba(245, 158, 11, 1)', borderWidth: 1, borderRadius: 4
                },
                {
                    label: 'F1 Score', data: f1,
                    backgroundColor: 'rgba(16, 185, 129, 0.6)', borderColor: 'rgba(16, 185, 129, 1)', borderWidth: 1, borderRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#94a3b8' } },
                title: { display: true, text: 'Per-Expert Performance (Precision / Recall / F1)', color: '#e2e8f0', font: { size: 14 } },
                tooltip: {
                    backgroundColor: 'rgba(15, 17, 21, 0.9)',
                    callbacks: { label: (c) => `${c.dataset.label}: ${(c.raw * 100).toFixed(1)}%` }
                }
            },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                y: { 
                    grid: { color: 'rgba(255,255,255,0.05)' }, 
                    min: 0, max: 1.0, 
                    ticks: { 
                        color: '#94a3b8', 
                        callback: (v) => (v * 100).toFixed(0) + '%',
                        stepSize: 0.2
                    }
                }
            }
        }
    });
}

function renderImportanceChart(moeImp, singleImp) {
    const ctx = document.getElementById('importanceChart').getContext('2d');
    if (importanceChart) importanceChart.destroy();

    const factors = ['Return_Trend', 'Momentum_Factor', 'Composite_Volatility', 'Volume_Shock', 'Latent_State_C', 'Tail_Kurtosis'];
    const shortLabels = ['Return\nTrend', 'Momentum\nFactor', 'Composite\nVolatility', 'Volume\nShock', 'Latent\nState', 'Tail\nKurtosis'];

    const lowData = factors.map(f => moeImp.Low?.[f] || 0);
    const medData = factors.map(f => moeImp.Medium?.[f] || 0);
    const highData = factors.map(f => moeImp.High?.[f] || 0);
    const singleData = factors.map(f => singleImp[f] || 0);

    importanceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: shortLabels,
            datasets: [
                {
                    label: 'E₀ (Low Vol)', data: lowData,
                    backgroundColor: 'rgba(16, 185, 129, 0.6)', borderColor: 'rgba(16, 185, 129, 1)', borderWidth: 1, borderRadius: 3
                },
                {
                    label: 'E₁ (Med Vol)', data: medData,
                    backgroundColor: 'rgba(245, 158, 11, 0.6)', borderColor: 'rgba(245, 158, 11, 1)', borderWidth: 1, borderRadius: 3
                },
                {
                    label: 'E₂ (High Vol)', data: highData,
                    backgroundColor: 'rgba(239, 68, 68, 0.6)', borderColor: 'rgba(239, 68, 68, 1)', borderWidth: 1, borderRadius: 3
                },
                {
                    label: 'Single RF (Baseline)', data: singleData,
                    backgroundColor: 'rgba(148, 163, 184, 0.4)', borderColor: 'rgba(148, 163, 184, 0.8)', borderWidth: 1,
                    borderRadius: 3, borderDash: [3, 3]
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#94a3b8', font: { size: 11 } } },
                title: { display: true, text: 'Normalized Feature Importance: Each Expert vs Single Model', color: '#e2e8f0', font: { size: 14 } },
                tooltip: {
                    backgroundColor: 'rgba(15, 17, 21, 0.9)',
                    callbacks: { label: (c) => `${c.dataset.label}: ${(c.raw * 100).toFixed(1)}%` }
                }
            },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8', font: { size: 10 } } },
                y: { 
                    grid: { color: 'rgba(255,255,255,0.05)' }, 
                    ticks: { color: '#94a3b8', callback: (v) => (v * 100) + '%' },
                    min: 0
                }
            }
        }
    });
}
