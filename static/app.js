// Smart Document Transformation - Frontend Application
// Supports true async parallel processing with real-time progress updates via SSE

let currentMode = 'single';
let currentDocumentTab = 'before';
let pairCount = 1;
let lastResult = null;

// Track multiple active jobs for parallel processing
const activeJobs = new Map(); // jobId -> { eventSource, status, filename, progress }

// Configuration
const USE_ASYNC = true;  // Always use async for parallel support

// ============================================================================
// Mode Selection
// ============================================================================

function setMode(mode) {
    currentMode = mode;
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });

    document.getElementById('singleExampleCard').style.display = mode === 'single' ? 'block' : 'none';
    document.getElementById('multiExampleCard').style.display = mode === 'multi' ? 'block' : 'none';

    // Update required attributes
    const singleInputs = document.querySelectorAll('#singleExampleCard input[type="file"]');
    const multiInputs = document.querySelectorAll('#multiExampleCard input[type="file"]');

    singleInputs.forEach(input => input.required = mode === 'single');
    multiInputs.forEach(input => input.required = mode === 'multi');
}

function addExamplePair() {
    pairCount++;
    const container = document.getElementById('examplePairs');
    const pairHtml = `
        <div class="example-pair" data-pair="${pairCount}">
            <button type="button" class="remove-pair-btn" onclick="removeExamplePair(${pairCount})">Remove</button>
            <div class="file-input-group">
                <div class="file-input-wrapper">
                    <label>Example Input ${pairCount} (Before)</label>
                    <input type="file" class="file-input" name="example_inputs" accept=".pdf,.docx,.txt,.md,.json,.xml,.csv">
                </div>
                <div class="file-input-wrapper">
                    <label>Example Output ${pairCount} (After)</label>
                    <input type="file" class="file-input" name="example_outputs" accept=".pdf,.docx,.txt,.md,.json,.xml,.csv">
                </div>
            </div>
        </div>
    `;
    container.insertAdjacentHTML('beforeend', pairHtml);
}

function removeExamplePair(pairNum) {
    const pair = document.querySelector(`.example-pair[data-pair="${pairNum}"]`);
    if (pair) pair.remove();
}

// ============================================================================
// Document Tabs (Before/After/Changes)
// ============================================================================

function setDocumentTab(tab) {
    currentDocumentTab = tab;
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tab);
    });

    document.getElementById('beforeTab').style.display = tab === 'before' ? 'block' : 'none';
    document.getElementById('afterTab').style.display = tab === 'after' ? 'block' : 'none';
    document.getElementById('changesTab').style.display = tab === 'changes' ? 'block' : 'none';

    // Render changes tab content when selected
    if (tab === 'changes' && lastResult) {
        renderChangesTab();
    }
}

function renderChangesTab() {
    if (!lastResult || !lastResult.diff_data) {
        document.getElementById('diffContent').innerHTML = '<p class="no-diff">No comparison data available</p>';
        return;
    }

    const diffData = lastResult.diff_data;
    const fullChangeNotice = document.getElementById('fullChangeNotice');
    const diffContent = document.getElementById('diffContent');

    if (diffData.is_full_change) {
        fullChangeNotice.style.display = 'flex';
        diffContent.innerHTML = '<p class="no-diff">Document was fully transformed. Compare using Before/After tabs.</p>';
    } else {
        fullChangeNotice.style.display = 'none';
        renderLineDiff(diffData.line_diff);
    }
}

function renderLineDiff(lineDiff) {
    const diffContent = document.getElementById('diffContent');
    if (!lineDiff || lineDiff.length === 0) {
        diffContent.innerHTML = '<p class="no-diff">No changes detected</p>';
        return;
    }

    let html = '<div class="diff-lines">';

    for (const op of lineDiff) {
        switch (op.type) {
            case 'equal':
                html += `<div class="diff-line equal"><span class="line-marker"> </span><span class="line-content">${escapeHtml(op.content)}</span></div>`;
                break;

            case 'insert':
                html += `<div class="diff-line insert"><span class="line-marker">+</span><span class="line-content">${escapeHtml(op.content)}</span></div>`;
                break;

            case 'delete':
                html += `<div class="diff-line delete"><span class="line-marker">-</span><span class="line-content">${escapeHtml(op.content)}</span></div>`;
                break;

            case 'replace':
                if (op.old && Array.isArray(op.old)) {
                    for (const line of op.old) {
                        html += `<div class="diff-line delete"><span class="line-marker">-</span><span class="line-content">${escapeHtml(line)}</span></div>`;
                    }
                }
                if (op.new && Array.isArray(op.new)) {
                    for (const line of op.new) {
                        html += `<div class="diff-line insert"><span class="line-marker">+</span><span class="line-content">${escapeHtml(line)}</span></div>`;
                    }
                }
                break;
        }
    }

    html += '</div>';
    diffContent.innerHTML = html;
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function updateDiffStats(diffData) {
    const statsBar = document.getElementById('diffStatsBar');
    if (!diffData) {
        statsBar.innerHTML = '';
        return;
    }

    const similarityPercent = Math.round(diffData.similarity_ratio * 100);
    let html = `<span class="stat-item">Similarity: <strong>${similarityPercent}%</strong></span>`;

    if (!diffData.is_full_change && diffData.stats) {
        html += `
            <span class="stat-item added">+${diffData.stats.lines_added} added</span>
            <span class="stat-item deleted">-${diffData.stats.lines_deleted} removed</span>
            ${diffData.stats.lines_modified > 0 ? `<span class="stat-item modified">~${diffData.stats.lines_modified} modified</span>` : ''}
        `;
    }

    statsBar.innerHTML = html;
}

// ============================================================================
// Active Jobs Panel (Parallel Processing)
// ============================================================================

function updateActiveJobsPanel() {
    const card = document.getElementById('activeJobsCard');
    const list = document.getElementById('activeJobsList');
    const count = document.getElementById('activeJobsCount');

    if (activeJobs.size === 0) {
        card.style.display = 'none';
        return;
    }

    card.style.display = 'block';
    count.textContent = activeJobs.size;

    let html = '';
    activeJobs.forEach((job, jobId) => {
        const statusClass = job.status === 'failed' ? 'error' : job.status === 'completed' ? 'success' : 'processing';
        html += `
            <div class="active-job-item ${statusClass}" data-job-id="${jobId}">
                <div class="job-info">
                    <span class="job-filename">${escapeHtml(job.filename)}</span>
                    <span class="job-status">${job.message || job.status}</span>
                </div>
                <div class="job-progress-bar">
                    <div class="job-progress-fill" style="width: ${job.progress}%"></div>
                </div>
                ${job.status === 'completed' ? `<button class="view-job-btn" onclick="viewCompletedJob('${jobId}')">View</button>` : ''}
            </div>
        `;
    });

    list.innerHTML = html;
}

function addActiveJob(jobId, filename) {
    activeJobs.set(jobId, {
        eventSource: null,
        status: 'pending',
        filename: filename,
        progress: 0,
        message: 'Starting...'
    });
    updateActiveJobsPanel();
}

function updateActiveJob(jobId, updates) {
    const job = activeJobs.get(jobId);
    if (job) {
        Object.assign(job, updates);
        updateActiveJobsPanel();
    }
}

function removeActiveJob(jobId) {
    const job = activeJobs.get(jobId);
    if (job && job.eventSource) {
        job.eventSource.close();
    }
    activeJobs.delete(jobId);
    updateActiveJobsPanel();
}

// ============================================================================
// Async Transformation with Parallel Support
// ============================================================================

async function startAsyncTransformation(formData, endpoint, filename) {
    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to start transformation');
        }

        const jobInfo = await response.json();
        const jobId = jobInfo.job_id;

        // Add to active jobs
        addActiveJob(jobId, filename);

        // Connect to SSE stream
        const eventSource = new EventSource(`/jobs/${jobId}/stream`);

        const job = activeJobs.get(jobId);
        if (job) {
            job.eventSource = eventSource;
        }

        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);

            updateActiveJob(jobId, {
                status: data.status,
                progress: data.progress,
                message: data.message
            });

            if (data.final) {
                eventSource.close();

                if (data.status === 'completed') {
                    updateActiveJob(jobId, { status: 'completed', message: 'Completed!' });
                    // Auto-load the first completed job
                    if (!lastResult) {
                        viewCompletedJob(jobId);
                    }
                } else {
                    updateActiveJob(jobId, { status: 'failed', message: data.error || 'Failed' });
                }

                // Remove from active jobs after a delay
                setTimeout(() => removeActiveJob(jobId), 3000);
            }
        };

        eventSource.onerror = () => {
            eventSource.close();
            updateActiveJob(jobId, { status: 'failed', message: 'Connection lost' });
            setTimeout(() => removeActiveJob(jobId), 3000);
        };

        return jobId;

    } catch (err) {
        throw err;
    }
}

async function viewCompletedJob(jobId) {
    try {
        const response = await fetch(`/jobs/${jobId}`);
        if (!response.ok) throw new Error('Failed to load job');

        const job = await response.json();
        displayResult(job);
    } catch (err) {
        showError(err.message);
    }
}

// ============================================================================
// Form Submission
// ============================================================================

document.getElementById('transformForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const error = document.getElementById('error');
    error.classList.remove('visible');

    try {
        const formData = new FormData();
        const newDocument = document.querySelector('input[name="new_document"]').files[0];

        if (!newDocument) {
            throw new Error('Please select a document to transform');
        }

        formData.append('new_document', newDocument);

        let endpoint;

        if (currentMode === 'single') {
            const exampleInput = document.querySelector('#singleExampleCard input[name="example_input"]').files[0];
            const exampleOutput = document.querySelector('#singleExampleCard input[name="example_output"]').files[0];

            if (!exampleInput || !exampleOutput) {
                throw new Error('Please select both example input and output documents');
            }

            formData.append('example_input', exampleInput);
            formData.append('example_output', exampleOutput);
            endpoint = '/transform-async';

        } else {
            // Multi-example mode - use sync endpoint for now
            endpoint = '/transform-multi';
            const inputs = document.querySelectorAll('#multiExampleCard input[name="example_inputs"]');
            const outputs = document.querySelectorAll('#multiExampleCard input[name="example_outputs"]');

            let hasValidPair = false;
            inputs.forEach((input, i) => {
                if (input.files[0] && outputs[i]?.files[0]) {
                    formData.append('example_inputs', input.files[0]);
                    formData.append('example_outputs', outputs[i].files[0]);
                    hasValidPair = true;
                }
            });

            if (!hasValidPair) {
                throw new Error('Please provide at least one complete example pair');
            }

            // Multi-example uses sync - display result directly
            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Transformation failed');
            }

            const result = await response.json();
            displayResult(result);
            loadHistory();
            return;
        }

        // Single example mode - use async for parallel processing
        await startAsyncTransformation(formData, endpoint, newDocument.name);

        // Clear the form for next upload
        document.getElementById('transformForm').reset();
        setMode(currentMode); // Re-apply mode to fix required attributes

    } catch (err) {
        showError(err.message);
    }
});

// ============================================================================
// Display Results
// ============================================================================

function displayResult(result) {
    lastResult = result;

    document.getElementById('jobId').textContent = result.job_id;
    document.getElementById('analysis').textContent = result.transformation_analysis;

    // Populate Before/After tabs
    document.getElementById('beforeContent').textContent = result.original_document || '';
    document.getElementById('afterContent').textContent = result.transformed_document || '';

    // Update diff stats
    updateDiffStats(result.diff_data);

    const metadata = result.metadata;
    document.getElementById('metadata').innerHTML = `
        <div class="metadata-item">Type: <span>${metadata.transformation_type}</span></div>
        <div class="metadata-item">Input Length: <span>${metadata.input_length.toLocaleString()}</span></div>
        <div class="metadata-item">Output Length: <span>${metadata.output_length.toLocaleString()}</span></div>
        ${metadata.chunks_processed > 1 ? `<div class="metadata-item">Chunks: <span>${metadata.chunks_processed}</span></div>` : ''}
        ${metadata.example_pairs_used ? `<div class="metadata-item">Examples Used: <span>${metadata.example_pairs_used}</span></div>` : ''}
        ${metadata.processing_mode ? `<div class="metadata-item">Mode: <span>${metadata.processing_mode}</span></div>` : ''}
    `;

    // Reset to "before" tab and show results
    setDocumentTab('before');

    const results = document.getElementById('results');
    results.classList.add('visible');
    results.scrollIntoView({ behavior: 'smooth' });

    // Refresh history
    loadHistory();
}

function showError(message) {
    const error = document.getElementById('error');
    error.textContent = message;
    error.classList.add('visible');
}

// ============================================================================
// Utility Functions
// ============================================================================

function copyToClipboard(elementId) {
    const text = document.getElementById(elementId).textContent;
    navigator.clipboard.writeText(text).then(() => {
        const btn = event.target;
        const originalText = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => btn.textContent = originalText, 1500);
    });
}

function downloadResult() {
    if (!lastResult) return;

    const blob = new Blob([lastResult.transformed_document], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `transformed_${lastResult.job_id || 'document'}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// ============================================================================
// History
// ============================================================================

async function loadHistory() {
    const historyList = document.getElementById('historyList');

    try {
        const response = await fetch('/jobs');
        if (!response.ok) throw new Error('Failed to load history');

        const data = await response.json();
        const jobs = data.jobs;

        if (jobs.length === 0) {
            historyList.innerHTML = '<p class="history-empty">No transformations yet. Run your first transformation above!</p>';
            return;
        }

        historyList.innerHTML = jobs.map(job => `
            <div class="history-item" onclick="viewJob('${job.job_id}')">
                <div class="history-item-header">
                    <span class="history-item-id">${job.job_id}</span>
                    <span class="history-item-date">${new Date(job.created_at).toLocaleString()}</span>
                </div>
                <div>
                    <span class="history-item-type">${job.metadata?.transformation_type || job.type}</span>
                    ${job.type === 'async' ? '<span class="history-item-badge">async</span>' : ''}
                </div>
            </div>
        `).join('');

    } catch (err) {
        historyList.innerHTML = '<p class="info-text">Failed to load history</p>';
    }
}

async function viewJob(jobId) {
    const error = document.getElementById('error');

    try {
        const response = await fetch(`/jobs/${jobId}`);
        if (!response.ok) throw new Error('Failed to load job');

        const job = await response.json();
        displayResult(job);

    } catch (err) {
        showError(err.message);
    }
}

// ============================================================================
// Cleanup and Initialization
// ============================================================================

window.addEventListener('beforeunload', () => {
    // Close all active EventSources
    activeJobs.forEach((job) => {
        if (job.eventSource) {
            job.eventSource.close();
        }
    });
});

// Load history on page load
document.addEventListener('DOMContentLoaded', loadHistory);
