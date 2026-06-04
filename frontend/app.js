const API_BASE_URL = 'http://localhost:8000/api/timeline';

let globalData = [];
let chartInstance = null;
let radarInstance = null;
let gatingInstance = null;

// UI Elements
const els = {
    dataset: document.getElementById('dataset-select'),
    qThresh: document.getElementById('q-threshold'),
    qVal: document.getElementById('q-value'),
    volLow: document.getElementById('vol-low'),
    volHigh: document.getElementById('vol-high'),
    loading: document.getElementById('chart-loading'),
    title: document.getElementById('timeline-title'),
    mRecall: document.getElementById('metric-recall'),
    mPrecision: document.getElementById('metric-precision'),
    mF1: document.getElementById('metric-f1'),
    srcLive: document.getElementById('src-live'),
    srcJournal: document.getElementById('src-journal'),
    journalNotice: document.getElementById('journal-notice')
};

let useJournal = false;

document.addEventListener('DOMContentLoaded', () => {
    // Initial fetch
    fetchData();

    // Event Listeners for dynamic updates
    els.dataset.addEventListener('change', fetchData);
    
    // For range slider, update label on input, but fetch only on 'change' (mouse up) to avoid spamming
    els.qThresh.addEventListener('input', (e) => {
        els.qVal.textContent = e.target.value;
    });
    els.qThresh.addEventListener('change', fetchData);
    
    els.volLow.addEventListener('change', fetchData);
    els.volHigh.addEventListener('change', fetchData);

    // Modal Control
    const archBtn = document.getElementById('btn-architecture');
    const archModal = document.getElementById('arch-modal');
    const closeBtn = document.getElementById('close-modal');
    
    if (archBtn && archModal) {
        archBtn.addEventListener('click', () => {
            archModal.classList.remove('hidden');
            archModal.classList.add('active');
        });
        closeBtn.addEventListener('click', () => {
            archModal.classList.add('hidden');
            archModal.classList.remove('active');
        });
    }

    // Toggle logic
    els.srcLive.addEventListener('click', () => setSource(false));
    els.srcJournal.addEventListener('click', () => setSource(true));

    function setSource(isJournal) {
        useJournal = isJournal;
        
        // Update UI
        if (useJournal) {
            els.srcJournal.classList.add('active');
            els.srcJournal.style.background = 'var(--accent)';
            els.srcJournal.style.color = 'white';
            els.srcLive.classList.remove('active');
            els.srcLive.style.background = 'transparent';
            els.srcLive.style.color = 'var(--text-muted)';
            els.journalNotice.style.display = 'block';
            // Dataset stays ENABLED — each window has different abstract metrics
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

        
        fetchData();
    }
});

async function fetchData() {
    // Show Loading
    els.loading.classList.add('active');
    
    // Construct Query URL
    const params = new URLSearchParams({
        dataset: els.dataset.value,
        q_threshold: els.qThresh.value,
        vol_q_low: els.volLow.value,
        vol_q_high: els.volHigh.value,
        use_journal: useJournal
    });
    
    // Update Title logic based on dataset
    let dsText = els.dataset.options[els.dataset.selectedIndex].text;
    els.title.textContent = `Performance Timeline - ${dsText}`;

    try {
        const res = await fetch(`${API_BASE_URL}?${params.toString()}`);
        const json = await res.json();
        
        if (json.status === 'success') {
            globalData = json.data.timeline;
            
            // Update Metrics
            const m = json.data.metrics;
            els.mRecall.textContent = (m.recall * 100).toFixed(1) + '%';
            els.mPrecision.textContent = (m.precision * 100).toFixed(1) + '%';
            els.mF1.textContent = (m.f1 * 100).toFixed(1) + '%';
            
            // Color coding based on F1 performance
            els.mF1.style.color = m.f1 > 0.8 ? 'var(--success)' : (m.f1 > 0.5 ? 'var(--warning)' : 'var(--danger)');
            
            // Render Charts
            if (chartInstance) {
                chartInstance.destroy();
            }
            initChart(globalData);
            initGatingChart(globalData);
            populateTable(globalData);
            
            // Hide Panel
            document.getElementById('explain-panel').classList.add('hidden');
        } else {
            console.error('API Error:', json.message);
            alert("Error fetching data: " + json.message);
        }
    } catch (err) {
        console.error('Fetch error:', err);
        // Only alert if it's not simply aborting/overlapping
        alert("Make sure the FastAPI backend is running on port 8000.");
    } finally {
        // Hide Loading
        els.loading.classList.remove('active');
    }
}

function initChart(data) {
    const ctx = document.getElementById('timelineChart').getContext('2d');
    
    // Create professional gradient for the primary line
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, 'rgba(59, 130, 246, 0.3)'); // Soft blue glow top
    gradient.addColorStop(1, 'rgba(59, 130, 246, 0.0)'); // Fade to transparent

    const labels = data.map(d => d.date);
    const prices = data.map(d => d.price);
    
    const config = {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'S&P 500 IndexProxy',
                    data: prices,
                    borderColor: 'rgba(59, 130, 246, 0.8)', // Brighter blue line
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.3, // Smoother curve
                    fill: true,
                    backgroundColor: gradient
                },
                {
                    type: 'bubble',
                    label: 'Detected Anomaly',
                    data: data.map((d, i) => d.is_anomaly ? {x: d.date, y: d.price, r: 6} : null).filter(Boolean),
                    backgroundColor: 'rgba(239, 68, 68, 0.8)',
                    borderColor: 'rgba(239, 68, 68, 1)',
                    borderWidth: 2,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    labels: { color: '#94a3b8' }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 17, 21, 0.9)',
                    titleColor: '#e2e8f0',
                    bodyColor: '#e2e8f0',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    callbacks: {
                        label: function(context) {
                            if (context.dataset.type === 'bubble') {
                                return `🔴 Anomaly Detected (${data[context.dataIndex]?.regime} Regime)`;
                            }
                            return `Price Index: ${context.parsed.y.toFixed(2)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8', maxTicksLimit: 12 }
                },
                y: {
                    type: 'logarithmic',
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8' }
                }
            },
            onClick: (e) => {
                const points = chartInstance.getElementsAtEventForMode(e, 'nearest', { intersect: true }, true);
                if (points.length) {
                    const basePoints = chartInstance.getElementsAtEventForMode(e, 'index', { intersect: false }, true);
                    if(basePoints.length) {
                         showExplanation(globalData[basePoints[0].index]);
                    }
                }
            }
        },
        plugins: [{
            id: 'regimeBackground',
            beforeDraw: (chart) => {
                const ctx = chart.ctx;
                const xAxis = chart.scales.x;
                const yAxis = chart.scales.y;
                const meta = chart.getDatasetMeta(0); // line dataset
                
                if(!globalData || !globalData.length || !meta.data || !meta.data.length) return;
                
                ctx.save();
                
                // Use actual pixel positions from the rendered data points
                let lastRegime = globalData[0].regime;
                let startX = meta.data[0] ? meta.data[0].x : xAxis.left;
                
                for(let i = 1; i < globalData.length; i++) {
                    let currentRegime = globalData[i].regime;
                    if(currentRegime !== lastRegime || i === globalData.length - 1) {
                        let endX = (meta.data[i]) ? meta.data[i].x : xAxis.right;
                        
                        if (lastRegime === 'High') {
                            ctx.fillStyle = 'rgba(239, 68, 68, 0.08)';
                        } else if (lastRegime === 'Medium') {
                            ctx.fillStyle = 'rgba(245, 158, 11, 0.05)';
                        } else {
                            ctx.fillStyle = 'rgba(16, 185, 129, 0.06)';
                        }
                        
                        ctx.fillRect(startX, yAxis.top, endX - startX, yAxis.bottom - yAxis.top);
                        
                        startX = endX;
                        lastRegime = currentRegime;
                    }
                }
                ctx.restore();
            }
        }]
    };

    chartInstance = new Chart(ctx, config);
}

function initGatingChart(data) {
    const ctx = document.getElementById('gatingChart').getContext('2d');
    
    if (gatingInstance) {
        gatingInstance.destroy();
    }
    
    const labels = data.map(d => d.date);
    const vols = data.map(d => d.volatility);
    const qLows = data.map(d => d.q_low);
    const qHighs = data.map(d => d.q_high);
    
    const config = {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Estimated Volatility (σt)',
                    data: vols,
                    borderColor: 'rgba(59, 130, 246, 0.8)',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.1,
                    fill: false
                },
                {
                    label: 'Dynamic Low Vol Threshold (q-low)',
                    data: qLows,
                    borderColor: 'rgba(16, 185, 129, 0.6)', // Fainter green
                    borderDash: [5, 5],
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.1
                },
                {
                    label: 'Dynamic High Vol Threshold (q-high)',
                    data: qHighs,
                    borderColor: 'rgba(239, 68, 68, 0.6)', // Fainter red
                    borderDash: [5, 5],
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    labels: { color: '#94a3b8', font: {size: 11} }
                },
                title: {
                    display: true,
                    text: 'Regime Gating Mechanics (Volatility vs Computed Quantile Thresholds)',
                    color: '#e2e8f0',
                    font: { size: 14 }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8', maxTicksLimit: 12 }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8' }
                }
            }
        }
    };
    gatingInstance = new Chart(ctx, config);
}

function showExplanation(dataPoint) {
    const indicator = document.getElementById('regime-indicator');
    indicator.textContent = `${dataPoint.regime} Volatility`;
    indicator.className = `indicator ${dataPoint.regime.toLowerCase()}`;
    
    const panel = document.getElementById('explain-panel');
    const driverList = document.getElementById('driver-list');
    
    if (dataPoint.is_anomaly && dataPoint.explanation) {
        panel.classList.remove('hidden');
        driverList.innerHTML = `
            <li><span style="color:#3b82f6;">${dataPoint.explanation.driver1.replace('_', ' ')}</span> (Primary)</li>
            <li><span style="color:#94a3b8;">${dataPoint.explanation.driver2.replace('_', ' ')}</span> (Secondary)</li>
        `;
        
        // Render Radar Chart
        if (radarInstance) radarInstance.destroy();
        const rtx = document.getElementById('radarChart').getContext('2d');
        radarInstance = new Chart(rtx, {
            type: 'radar',
            data: {
                labels: ['Return', 'Momentum', 'Volatility', 'Volume', 'Kurtosis'],
                datasets: [{
                    label: 'Factor Signature',
                    data: dataPoint.explanation.radar,
                    backgroundColor: 'rgba(59, 130, 246, 0.2)',
                    borderColor: 'rgba(59, 130, 246, 1)',
                    pointBackgroundColor: 'rgba(59, 130, 246, 1)',
                    borderWidth: 2
                }]
            },
            options: {
                scales: {
                    r: {
                        angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        pointLabels: { color: '#e2e8f0', font: {size: 10} },
                        ticks: { display: false, min: 0 }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: 'rgba(15, 17, 21, 0.9)',
                        callbacks: {
                            label: (ctx) => `${ctx.label}: ${ctx.raw.toFixed(2)}`
                        }
                    }
                },
                maintainAspectRatio: false
            }
        });
        
    } else {
        panel.classList.add('hidden');
    }
}

function populateTable(data) {
    const tbody = document.querySelector('#anomalyTable tbody');
    tbody.innerHTML = '';
    
    const anomalies = data.filter(d => d.is_anomaly).reverse().slice(0, 50);
    
    for (const item of anomalies) {
        const tr = document.createElement('tr');
        
        const f1 = item.explanation ? item.explanation.driver1 : 'N/A';
        const f2 = item.explanation ? item.explanation.driver2 : 'N/A';
        
        tr.innerHTML = `
            <td>${item.date}</td>
            <td><span style="color:var(--warning)">${item.regime}</span></td>
            <td><span class="badge-danger">ANOMALY</span></td>
            <td>${f1.replace('_', ' ')}</td>
            <td>${f2.replace('_', ' ')}</td>
        `;
        
        tr.addEventListener('click', () => {
             showExplanation(item);
        });
        
        tbody.appendChild(tr);
    }
}
