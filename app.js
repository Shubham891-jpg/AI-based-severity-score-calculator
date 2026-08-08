// IT Ticket Severity Calculator Dashboard Controller

const DEFAULT_BACKEND_URL = "https://ai-based-severity-score-calculator.onrender.com";

function getApiBaseUrl() {
    const saved = localStorage.getItem('severity-api-url');
    if (saved && saved.trim()) return saved.trim().replace(/\/$/, "");
    
    const host = window.location.hostname;
    const protocol = window.location.protocol;
    
    // Auto-detect Hugging Face Spaces or static hosting
    if (host.includes('hf.space') || host.includes('huggingface.co') || protocol === 'file:' || (host === 'localhost' && window.location.port !== '8000' && window.location.port !== '10000')) {
        return DEFAULT_BACKEND_URL;
    }
    return window.location.origin;
}

let API_BASE_URL = getApiBaseUrl();

// Application State
let state = {
    activeTab: 'single',
    theme: 'dark',
    localHistory: [],
    batchResults: [],
    modelInfo: null,
    apiStatus: 'unknown'
};

// Preset Tickets (Professional cases)
const PRESETS = [
    {
        id: 1,
        severity: 'high',
        label: 'High Severity',
        text: 'All servers are down, complete system failure, no one can work',
        desc: 'Global Outage'
    },
    {
        id: 2,
        severity: 'medium',
        label: 'Medium Severity',
        text: 'Database is extremely slow, all applications timing out',
        desc: 'Performance degradation'
    },
    {
        id: 3,
        severity: 'low',
        label: 'Low Severity',
        text: 'Office printer is not working, affecting multiple users',
        desc: 'Localized Hardware issue'
    },
    {
        id: 4,
        severity: 'low',
        label: 'Low Severity',
        text: 'User needs password reset for their account',
        desc: 'Standard Identity request'
    },
    {
        id: 5,
        severity: 'high',
        label: 'Hindi High',
        text: 'सर्वर डाउन है और कोई भी काम नहीं कर सकता',
        desc: 'Hindi Support Outage'
    }
];

// Initialize on Page Load
window.addEventListener('DOMContentLoaded', () => {
    initApp();
});

// App Entry Point
function initApp() {
    loadSettings();
    renderPresets();
    setupEventListeners();
    checkApiHealth();
    fetchDiagnostics();
}

// Load Settings from LocalStorage
function loadSettings() {
    // Theme
    const savedTheme = localStorage.getItem('severity-theme');
    if (savedTheme === 'light') {
        state.theme = 'light';
        document.body.classList.add('light-theme');
        const themeIcon = document.querySelector('#themeToggle i');
        if (themeIcon) {
            themeIcon.className = 'fa-solid fa-moon';
        }
    } else {
        state.theme = 'dark';
        document.body.classList.remove('light-theme');
        const themeIcon = document.querySelector('#themeToggle i');
        if (themeIcon) {
            themeIcon.className = 'fa-solid fa-sun';
        }
    }
    
    // History
    const savedHistory = localStorage.getItem('severity-history');
    if (savedHistory) {
        try {
            state.localHistory = JSON.parse(savedHistory);
        } catch (e) {
            state.localHistory = [];
        }
    }
    
    updateAnalyticsCharts();
}

// Save History to LocalStorage
function saveHistory() {
    localStorage.setItem('severity-history', JSON.stringify(state.localHistory));
    updateAnalyticsCharts();
}

// Render Preset Cards
function renderPresets() {
    const presetsContainer = document.getElementById('presetsGrid');
    if (!presetsContainer) return;
    
    presetsContainer.innerHTML = PRESETS.map(preset => `
        <div class="preset-card" onclick="selectPreset(${preset.id})">
            <span class="preset-header ${preset.severity}">${preset.label}</span>
            <div class="preset-text" title="${preset.text}">${preset.text}</div>
        </div>
    `).join('');
}

// Event Listeners Setup
function setupEventListeners() {
    // Tabs Navigation
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.getAttribute('data-tab');
            switchTab(tabName);
        });
    });
    
    // Theme Toggle
    const themeBtn = document.getElementById('themeToggle');
    if (themeBtn) {
        themeBtn.addEventListener('click', toggleTheme);
    }
    
    // Textarea character count
    const textarea = document.getElementById('ticketText');
    const charCount = document.getElementById('charCount');
    if (textarea && charCount) {
        textarea.addEventListener('input', () => {
            charCount.textContent = `${textarea.value.length}/5000`;
        });
        
        // Enter key submission
        textarea.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey) {
                e.preventDefault();
                predictSeverity();
            }
        });
    }
    
    // Batch file drag/drop
    const dropZone = document.getElementById('dragDropZone');
    const fileInput = document.getElementById('fileInput');
    if (dropZone && fileInput) {
        dropZone.addEventListener('click', () => fileInput.click());
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        });
        
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.style.borderColor = 'var(--text-primary)';
            dropZone.style.background = 'var(--bg-tertiary)';
        });
        
        dropZone.addEventListener('dragleave', () => {
            dropZone.style.borderColor = 'var(--border-color)';
            dropZone.style.background = 'var(--bg-primary)';
        });
        
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.style.borderColor = 'var(--border-color)';
            dropZone.style.background = 'var(--bg-primary)';
            if (e.dataTransfer.files.length > 0) {
                handleFileSelect(e.dataTransfer.files[0]);
            }
        });
    }
    
    // Search batch results
    const searchInput = document.getElementById('batchSearch');
    if (searchInput) {
        searchInput.addEventListener('input', filterBatchTable);
    }
}

// Switch Tabs
function switchTab(tabName) {
    state.activeTab = tabName;
    
    // Tab Buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        if (btn.getAttribute('data-tab') === tabName) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
    
    // Panels
    document.querySelectorAll('.panel').forEach(panel => {
        if (panel.id === `${tabName}Panel`) {
            panel.classList.add('active');
        } else {
            panel.classList.remove('active');
        }
    });
    
    // Diagnostics tab action
    if (tabName === 'diagnostics') {
        fetchDiagnostics();
        updateAnalyticsCharts();
    }
}

// Toggle Theme
function toggleTheme() {
    const themeIcon = document.querySelector('#themeToggle i');
    if (state.theme === 'dark') {
        state.theme = 'light';
        document.body.classList.add('light-theme');
        if (themeIcon) themeIcon.className = 'fa-solid fa-moon';
        localStorage.setItem('severity-theme', 'light');
    } else {
        state.theme = 'dark';
        document.body.classList.remove('light-theme');
        if (themeIcon) themeIcon.className = 'fa-solid fa-sun';
        localStorage.setItem('severity-theme', 'dark');
    }
    updateAnalyticsCharts();
}

// HTML Escaping Utility for XSS Prevention
function escapeHtml(text) {
    if (!text) return '';
    return String(text)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// Alert Toast Display
function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    if (!container) return;
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    let icon = 'fa-circle-info';
    if (type === 'error') icon = 'fa-circle-xmark';
    if (type === 'success') icon = 'fa-circle-check';
    
    toast.innerHTML = `
        <i class="fa-solid ${icon}"></i>
        <div>${message}</div>
    `;
    
    container.appendChild(toast);
    
    // Remove toast after 4s
    setTimeout(() => {
        toast.style.animation = 'fadeIn 100ms ease-out reverse';
        setTimeout(() => toast.remove(), 100);
    }, 4000);
}

// Check Server Health
async function checkApiHealth() {
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            state.apiStatus = 'healthy';
            if (statusDot) statusDot.className = 'status-dot active';
            if (statusText) statusText.textContent = 'Service Connected';
        } else {
            throw new Error();
        }
    } catch (error) {
        state.apiStatus = 'offline';
        if (statusDot) statusDot.className = 'status-dot inactive';
        if (statusText) statusText.textContent = 'Disconnected';
        showToast('Server connection offline. Check backend status.', 'error');
    }
}

// Select a Preset Card
function selectPreset(id) {
    const preset = PRESETS.find(p => p.id === id);
    if (!preset) return;
    
    const textarea = document.getElementById('ticketText');
    if (textarea) {
        textarea.value = preset.text;
        textarea.dispatchEvent(new Event('input'));
        predictSeverity();
    }
}

// Reset Single Analyzer
function clearSingleInput() {
    const textarea = document.getElementById('ticketText');
    if (textarea) {
        textarea.value = '';
        textarea.dispatchEvent(new Event('input'));
    }
    
    document.getElementById('resultPlaceholder').style.display = 'flex';
    document.getElementById('resultVisualizer').classList.remove('active');
}

// Call Prediction API (Single)
async function predictSeverity() {
    const textInput = document.getElementById('ticketText');
    if (!textInput) return;
    
    const text = textInput.value.trim();
    if (!text) {
        showToast('Please input ticket description.', 'error');
        return;
    }
    
    const button = document.getElementById('analyzeBtn');
    const placeholder = document.getElementById('resultPlaceholder');
    const visualizer = document.getElementById('resultVisualizer');
    const loader = document.getElementById('singleLoader');
    
    button.disabled = true;
    button.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Processing...';
    placeholder.style.display = 'none';
    visualizer.classList.remove('active');
    loader.style.display = 'flex';
    
    const startTime = Date.now();
    
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                ticket_text: text
            })
        });
        
        if (!response.ok) {
            let errorMsg = `Server error (${response.status})`;
            try {
                const errData = await response.json();
                errorMsg = errData.detail || errData.error || errorMsg;
            } catch (e) {
                try {
                    const rawText = await response.text();
                    if (rawText && rawText.trim()) {
                        const cleanText = rawText.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
                        if (cleanText) {
                            errorMsg = cleanText.substring(0, 150);
                        }
                    }
                } catch (tErr) {
                    // Ignore text read error
                }
            }
            throw new Error(errorMsg);
        }
        
        const result = await response.json();
        const duration = Date.now() - startTime;
        
        // Render Result
        renderSingleResult(result, duration);
        
        // Add to local history & save
        state.localHistory.unshift({
            text: text,
            score: result.severity_score,
            category: result.severity_category,
            language: result.detected_language || 'en',
            timestamp: new Date().toISOString()
        });
        
        if (state.localHistory.length > 50) {
            state.localHistory.pop();
        }
        
        saveHistory();
        
    } catch (error) {
        showToast(error.message || 'An error occurred while connecting to the server', 'error');
        placeholder.style.display = 'flex';
    } finally {
        button.disabled = false;
        button.innerHTML = '<i class="fa-solid fa-magnifying-glass"></i> Analyze Severity';
        loader.style.display = 'none';
    }
}

// Render Single Prediction Visuals
function renderSingleResult(result, duration) {
    document.getElementById('resultPlaceholder').style.display = 'none';
    const visualizer = document.getElementById('resultVisualizer');
    visualizer.classList.add('active');
    
    const scoreVal = result.severity_score;
    const cat = result.severity_category.toLowerCase();
    
    // Large score
    document.getElementById('scoreText').textContent = scoreVal.toFixed(1);
    
    // Badge Category
    const categoryBadge = document.getElementById('resCategory');
    categoryBadge.className = `badge ${cat}`;
    categoryBadge.textContent = result.severity_category;
    
    // Metric Linear Bar Fill width: range is 10 to 100 (diff is 90)
    const metricFill = document.getElementById('metricBarFill');
    if (metricFill) {
        const percent = ((scoreVal - 10) / 90) * 100;
        metricFill.className = `metric-bar-fill ${cat}`;
        metricFill.style.width = `${percent}%`;
    }
    
    // Confidence Mini Progress
    const confVal = result.confidence * 100;
    document.getElementById('resConfidence').textContent = `${confVal.toFixed(1)}%`;
    const confBar = document.getElementById('confidenceBarFill');
    if (confBar) {
        confBar.style.width = `${confVal}%`;
    }
    
    // Language
    let lang = 'English';
    if (result.detected_language === 'hi') lang = 'Hindi';
    if (result.detected_language === 'en') lang = 'English';
    document.getElementById('resLanguage').textContent = lang;
    
    // Latency Speed
    document.getElementById('resTime').textContent = `${duration}ms`;
    
    // Keywords Token Cloud
    const tokenCloud = document.getElementById('resTokens');
    if (tokenCloud) {
        if (result.processed_text) {
            const tokens = result.processed_text.split(/\s+/).filter(t => t.length > 0);
            if (tokens.length > 0) {
                tokenCloud.innerHTML = tokens.map(token => `<span class="token-tag">${escapeHtml(token)}</span>`).join('');
            } else {
                tokenCloud.innerHTML = `<span class="text-muted" style="font-size: 0.85em;">No features processed</span>`;
            }
        } else {
            tokenCloud.innerHTML = `<span class="text-muted" style="font-size: 0.85em;">No features processed</span>`;
        }
    }
}

// File Import Handler (Batch)
function handleFileSelect(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const text = e.target.result;
        const textarea = document.getElementById('batchInputText');
        if (textarea) {
            textarea.value = text;
            showToast(`Loaded ${file.name} list`, 'success');
        }
    };
    reader.readAsText(file);
}

// Process Batch Predictions
async function processBatch() {
    const textarea = document.getElementById('batchInputText');
    if (!textarea) return;
    
    const rawText = textarea.value.trim();
    if (!rawText) {
        showToast('Please input batch descriptions (one per line).', 'error');
        return;
    }
    
    const tickets = rawText.split('\n')
        .map(line => line.trim())
        .filter(line => line.length > 0);
        
    if (tickets.length === 0) {
        showToast('No valid ticket descriptions detected.', 'error');
        return;
    }
    
    if (tickets.length > 100) {
        showToast('FastAPI limits batch batches to 100 items. Truncated list.', 'info');
        tickets.length = 100;
    }
    
    const runBtn = document.getElementById('batchRunBtn');
    const resetBtn = document.getElementById('batchResetBtn');
    const exportBtn = document.getElementById('batchExportBtn');
    const progressContainer = document.getElementById('batchProgressContainer');
    const progressBarFill = document.getElementById('batchProgressBarFill');
    
    runBtn.disabled = true;
    resetBtn.disabled = true;
    exportBtn.disabled = true;
    progressContainer.style.display = 'block';
    progressBarFill.style.width = '25%';
    
    try {
        const response = await fetch(`${API_BASE_URL}/predict/batch`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                tickets: tickets
            })
        });
        
        progressBarFill.style.width = '75%';
        
        if (!response.ok) {
            let errorMsg = `Batch calculator failed (${response.status})`;
            try {
                const errData = await response.json();
                errorMsg = errData.detail || errData.error || errorMsg;
            } catch (e) {
                try {
                    const rawText = await response.text();
                    if (rawText && rawText.trim()) {
                        const cleanText = rawText.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
                        if (cleanText) {
                            errorMsg = cleanText.substring(0, 150);
                        }
                    }
                } catch (tErr) {
                    // Ignore text read error
                }
            }
            throw new Error(errorMsg);
        }
        
        const data = await response.json();
        progressBarFill.style.width = '100%';
        
        // Save results
        state.batchResults = data.predictions.map((pred, index) => ({
            index: index + 1,
            text: tickets[index] || '',
            score: pred.severity_score,
            category: pred.severity_category,
            confidence: pred.confidence,
            language: pred.detected_language || 'en'
        }));
        
        // Feed into local history for analytics updates
        state.batchResults.forEach(r => {
            state.localHistory.unshift({
                text: r.text,
                score: r.score,
                category: r.category,
                language: r.language,
                timestamp: new Date().toISOString()
            });
        });
        if (state.localHistory.length > 100) {
            state.localHistory.length = 100;
        }
        saveHistory();
        
        // Render table
        renderBatchTable(state.batchResults);
        showToast(`Successfully analyzed ${data.total_tickets} items`, 'success');
        exportBtn.disabled = false;
        
    } catch (error) {
        showToast(error.message, 'error');
    } finally {
        runBtn.disabled = false;
        resetBtn.disabled = false;
        
        setTimeout(() => {
            progressContainer.style.display = 'none';
            progressBarFill.style.width = '0%';
        }, 600);
    }
}

// Reset Batch Analyzer
function resetBatch() {
    const textarea = document.getElementById('batchInputText');
    const tableBody = document.getElementById('batchTableBody');
    const exportBtn = document.getElementById('batchExportBtn');
    
    if (textarea) textarea.value = '';
    if (tableBody) tableBody.innerHTML = `<tr><td colspan="6" class="text-muted" style="text-align: center; padding: 20px;">No batch results available. Run prediction.</td></tr>`;
    if (exportBtn) exportBtn.disabled = true;
    state.batchResults = [];
}

// Render Batch Table
function renderBatchTable(data) {
    const tableBody = document.getElementById('batchTableBody');
    if (!tableBody) return;
    
    if (data.length === 0) {
        tableBody.innerHTML = `<tr><td colspan="6" class="text-muted" style="text-align: center; padding: 20px;">No matches found</td></tr>`;
        return;
    }
    
    tableBody.innerHTML = data.map(item => {
        const catClass = item.category.toLowerCase();
        const confPercent = (item.confidence * 100).toFixed(1);
        let langLabel = item.language === 'hi' ? 'Hindi' : 'English';
        
        return `
            <tr>
                <td style="font-weight: 700; width: 40px; font-family: var(--font-mono);">${item.index}</td>
                <td title="${escapeHtml(item.text)}">${escapeHtml(item.text)}</td>
                <td><span class="score-badge ${catClass}">${item.score.toFixed(1)}</span></td>
                <td style="font-weight: 600;">${item.category}</td>
                <td>${confPercent}%</td>
                <td><span style="font-size: 0.9em; background: var(--bg-tertiary); padding: 2px 6px; border-radius: 4px;">${langLabel}</span></td>
            </tr>
        `;
    }).join('');
}

// Filter/Search Batch Table
function filterBatchTable() {
    const query = document.getElementById('batchSearch').value.toLowerCase().trim();
    if (!query) {
        renderBatchTable(state.batchResults);
        return;
    }
    
    const filtered = state.batchResults.filter(item => 
        item.text.toLowerCase().includes(query) ||
        item.category.toLowerCase().includes(query) ||
        item.language.toLowerCase().includes(query)
    );
    
    renderBatchTable(filtered);
}

// Sort Batch Table variables
let currentSort = { col: '', asc: true };

function sortBatchTable(column) {
    if (state.batchResults.length === 0) return;
    
    if (currentSort.col === column) {
        currentSort.asc = !currentSort.asc;
    } else {
        currentSort.col = column;
        currentSort.asc = true;
    }
    
    state.batchResults.sort((a, b) => {
        let valA = a[column];
        let valB = b[column];
        
        if (typeof valA === 'string') {
            return currentSort.asc ? valA.localeCompare(valB) : valB.localeCompare(valA);
        } else {
            return currentSort.asc ? valA - valB : valB - valA;
        }
    });
    
    renderBatchTable(state.batchResults);
    
    document.querySelectorAll('.data-table th i').forEach(icon => {
        icon.className = 'fa-solid fa-sort';
    });
    
    const thActive = document.querySelector(`.data-table th[data-sort="${column}"] i`);
    if (thActive) {
        thActive.className = currentSort.asc ? 'fa-solid fa-sort-up' : 'fa-solid fa-sort-down';
    }
}

// Export Batch Results to CSV
function exportBatchToCSV() {
    if (state.batchResults.length === 0) return;
    
    const headers = ['Index', 'Ticket Text', 'Severity Score', 'Severity Category', 'Confidence', 'Language'];
    const rows = state.batchResults.map(item => [
        item.index,
        `"${item.text.replace(/"/g, '""')}"`,
        item.score.toFixed(2),
        item.category,
        item.confidence.toFixed(4),
        item.language
    ]);
    
    const csvContent = "data:text/csv;charset=utf-8," 
        + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
        
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `severity_export_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    showToast('Successfully exported CSV', 'success');
}

// Fetch Diagnostics Metadata
async function fetchDiagnostics() {
    try {
        const response = await fetch(`${API_BASE_URL}/model/info`);
        if (response.ok) {
            const data = await response.json();
            state.modelInfo = data.model_info;
            renderDiagnosticsInfo();
        } else {
            const healthResp = await fetch(`${API_BASE_URL}/health`);
            if (healthResp.ok) {
                const data = await healthResp.json();
                state.modelInfo = data.model_info;
                renderDiagnosticsInfo();
            }
        }
    } catch (e) {
        renderDiagnosticsInfo();
    }
}

// Render Diagnostics Info Cards
function renderDiagnosticsInfo() {
    const typeVal = document.getElementById('diagModelType');
    const paramVal = document.getElementById('diagModelParams');
    const embedVal = document.getElementById('diagEmbeddings');
    const timeVal = document.getElementById('diagLoadTime');
    
    if (state.modelInfo) {
        if (typeVal) typeVal.textContent = state.modelInfo.model_type || 'RandomForestRegressor';
        
        if (paramVal) {
            const estimators = state.modelInfo.n_estimators || 100;
            const depth = state.modelInfo.max_depth || 'None';
            paramVal.textContent = `${estimators} estimators, max_depth: ${depth}`;
        }
        
        if (embedVal) {
            const name = state.modelInfo.embedding_model || 'paraphrase-multilingual';
            const dim = state.modelInfo.embedding_dim || 384;
            embedVal.textContent = `${name} (${dim}d)`;
        }
    } else {
        if (typeVal) typeVal.textContent = 'RandomForestRegressor';
        if (paramVal) paramVal.textContent = '100 estimators, max_depth: None';
        if (embedVal) embedVal.textContent = 'paraphrase-multilingual-MiniLM-L12-v2 (384d)';
    }
    
    if (timeVal) {
        timeVal.textContent = new Date().toLocaleTimeString();
    }
}

// Update Local History Analytics Chart (Flat monochrome bars)
function updateAnalyticsCharts() {
    const counts = { high: 0, medium: 0, low: 0 };
    state.localHistory.forEach(item => {
        const cat = item.category.toLowerCase();
        if (counts.hasOwnProperty(cat)) {
            counts[cat]++;
        }
    });
    
    const total = state.localHistory.length;
    
    const criticalVal = document.getElementById('statHighCount');
    const mediumVal = document.getElementById('statMediumCount');
    const lowVal = document.getElementById('statLowCount');
    const totalVal = document.getElementById('statTotalCount');
    
    if (criticalVal) criticalVal.textContent = counts.high;
    if (mediumVal) mediumVal.textContent = counts.medium;
    if (lowVal) lowVal.textContent = counts.low;
    if (totalVal) totalVal.textContent = total;
    
    const criticalBar = document.getElementById('barHighFill');
    const mediumBar = document.getElementById('barMediumFill');
    const lowBar = document.getElementById('barLowFill');
    
    if (total > 0) {
        if (criticalBar) criticalBar.style.width = `${(counts.high / total) * 100}%`;
        if (mediumBar) mediumBar.style.width = `${(counts.medium / total) * 100}%`;
        if (lowBar) lowBar.style.width = `${(counts.low / total) * 100}%`;
    } else {
        if (criticalBar) criticalBar.style.width = '0%';
        if (mediumBar) mediumBar.style.width = '0%';
        if (lowBar) lowBar.style.width = '0%';
    }
}

// Export functions to global scope for HTML inline calls
window.clearSingleInput = clearSingleInput;
window.predictSeverity = predictSeverity;
window.processBatch = processBatch;
window.resetBatch = resetBatch;
window.exportBatchToCSV = exportBatchToCSV;
window.sortBatchTable = sortBatchTable;
window.selectPreset = selectPreset;
