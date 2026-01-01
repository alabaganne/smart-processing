// Smart Document Transformation - Frontend Application
// Supports async processing with real-time progress updates via SSE

let currentMode = 'single';
let currentResultView = 'result';
let pairCount = 1;
let lastResult = null;
let currentEventSource = null;  // Track SSE connection

// Configuration
const USE_ASYNC = true;  // Use async processing for large documents
const LARGE_FILE_THRESHOLD = 50000;  // 50KB - use async for files larger than this

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
// Result View Toggle (Result Only vs Comparison)
// ============================================================================

function setResultView(view) {
    currentResultView = view;
    document.querySelectorAll('.view-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.view === view);
    });

    const resultOnlyView = document.getElementById('resultOnlyView');
    const comparisonView = document.getElementById('comparisonView');

    if (view === 'result') {
        resultOnlyView.style.display = 'block';
        comparisonView.style.display = 'none';
    } else {
        resultOnlyView.style.display = 'none';
        comparisonView.style.display = 'block';
        renderComparisonView();
    }
}

function renderComparisonView() {
    if (!lastResult || !lastResult.diff_data) {
        return;
    }

    const diffData = lastResult.diff_data;
    const fullChangeNotice = document.getElementById('fullChangeNotice');
    const sideBySideView = document.getElementById('sideBySideView');
    const diffView = document.getElementById('diffView');
    const diffStats = document.getElementById('diffStats');

    // Show similarity stats
    const similarityPercent = Math.round(diffData.similarity_ratio * 100);
    diffStats.innerHTML = `
        <span class="stat-item">Similarity: <strong>${similarityPercent}%</strong></span>
    `;

    if (diffData.is_full_change) {
        // Full document change - show side by side
        fullChangeNotice.style.display = 'flex';
        sideBySideView.style.display = 'flex';
        diffView.style.display = 'none';

        document.getElementById('originalDoc').textContent = lastResult.original_document || '';
        document.getElementById('transformedDoc').textContent = lastResult.transformed_document || '';
    } else {
        // Partial change - show diff with highlighting
        fullChangeNotice.style.display = 'none';
        sideBySideView.style.display = 'none';
        diffView.style.display = 'block';

        // Add stats
        if (diffData.stats) {
            diffStats.innerHTML += `
                <span class="stat-item added">+${diffData.stats.lines_added} added</span>
                <span class="stat-item deleted">-${diffData.stats.lines_deleted} removed</span>
                ${diffData.stats.lines_modified > 0 ? `<span class="stat-item modified">~${diffData.stats.lines_modified} modified</span>` : ''}
            `;
        }

        // Render the diff
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
                // Show old lines as deleted
                if (op.old && Array.isArray(op.old)) {
                    for (const line of op.old) {
                        html += `<div class="diff-line delete"><span class="line-marker">-</span><span class="line-content">${escapeHtml(line)}</span></div>`;
                    }
                }
                // Show new lines as inserted
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

// Progress UI functions
function showProgress() {
    const progressContainer = document.getElementById('progressContainer');
    if (progressContainer) {
        progressContainer.classList.add('visible');
    }
}

function hideProgress() {
    const progressContainer = document.getElementById('progressContainer');
    if (progressContainer) {
        progressContainer.classList.remove('visible');
    }
}

function updateProgress(progress, message, currentChunk = null, totalChunks = null) {
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const chunkInfo = document.getElementById('chunkInfo');

    if (progressBar) {
        progressBar.style.width = `${progress}%`;
        progressBar.setAttribute('aria-valuenow', progress);
    }

    if (progressText) {
        progressText.textContent = message || `Processing... ${progress}%`;
    }

    if (chunkInfo && currentChunk !== null && totalChunks !== null && totalChunks > 1) {
        chunkInfo.textContent = `Chunk ${currentChunk} of ${totalChunks}`;
        chunkInfo.style.display = 'block';
    } else if (chunkInfo) {
        chunkInfo.style.display = 'none';
    }
}

function updatePartialResult(preview) {
    const partialResult = document.getElementById('partialResult');
    if (partialResult && preview) {
        partialResult.textContent = preview;
        partialResult.parentElement.style.display = 'block';
    }
}

// Check if file is large enough to warrant async processing
function shouldUseAsync(files) {
    if (!USE_ASYNC) return false;

    for (const file of files) {
        if (file && file.size > LARGE_FILE_THRESHOLD) {
            return true;
        }
    }
    return false;
}

// Close any existing SSE connection
function closeEventSource() {
    if (currentEventSource) {
        currentEventSource.close();
        currentEventSource = null;
    }
}

// Handle async transformation with SSE
async function handleAsyncTransformation(formData, endpoint) {
    const submitBtn = document.getElementById('submitBtn');
    const loading = document.getElementById('loading');
    const error = document.getElementById('error');
    const results = document.getElementById('results');

    try {
        // Start the async job
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

        // Show progress UI
        loading.classList.remove('visible');
        showProgress();
        updateProgress(0, 'Starting transformation...', 0, jobInfo.total_chunks);

        // Connect to SSE stream
        return new Promise((resolve, reject) => {
            closeEventSource();
            currentEventSource = new EventSource(`/jobs/${jobId}/stream`);

            currentEventSource.onmessage = (event) => {
                const data = JSON.parse(event.data);

                // Update progress
                updateProgress(
                    data.progress,
                    data.message,
                    data.current_chunk,
                    data.total_chunks
                );

                // Show partial result if available
                if (data.partial_result_preview) {
                    updatePartialResult(data.partial_result_preview);
                }

                // Check if job is complete
                if (data.final) {
                    closeEventSource();
                    hideProgress();

                    if (data.status === 'completed') {
                        // Fetch the full result
                        fetchJobResult(jobId).then(resolve).catch(reject);
                    } else {
                        reject(new Error(data.error || 'Transformation failed'));
                    }
                }
            };

            currentEventSource.onerror = (err) => {
                closeEventSource();
                hideProgress();

                // Try to fetch result anyway (job might have completed)
                fetchJobResult(jobId)
                    .then(resolve)
                    .catch(() => reject(new Error('Connection lost. Please check job status.')));
            };
        });

    } catch (err) {
        hideProgress();
        throw err;
    }
}

// Fetch completed job result
async function fetchJobResult(jobId) {
    const response = await fetch(`/jobs/${jobId}`);
    if (!response.ok) {
        throw new Error('Failed to fetch job result');
    }
    return response.json();
}

// Handle synchronous transformation (original behavior)
async function handleSyncTransformation(formData, endpoint) {
    const response = await fetch(endpoint, {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Transformation failed');
    }

    return response.json();
}

document.getElementById('transformForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const submitBtn = document.getElementById('submitBtn');
    const loading = document.getElementById('loading');
    const error = document.getElementById('error');
    const results = document.getElementById('results');

    submitBtn.disabled = true;
    loading.classList.add('visible');
    error.classList.remove('visible');
    results.classList.remove('visible');
    hideProgress();

    try {
        const formData = new FormData();
        const newDocument = document.querySelector('input[name="new_document"]').files[0];

        if (!newDocument) {
            throw new Error('Please select a document to transform');
        }

        formData.append('new_document', newDocument);

        let endpoint;
        let useAsync = false;
        const filesToCheck = [newDocument];

        if (currentMode === 'single') {
            const exampleInput = document.querySelector('#singleExampleCard input[name="example_input"]').files[0];
            const exampleOutput = document.querySelector('#singleExampleCard input[name="example_output"]').files[0];

            if (!exampleInput || !exampleOutput) {
                throw new Error('Please select both example input and output documents');
            }

            formData.append('example_input', exampleInput);
            formData.append('example_output', exampleOutput);
            filesToCheck.push(exampleInput, exampleOutput);

            // Determine endpoint based on file size
            useAsync = shouldUseAsync(filesToCheck);
            endpoint = useAsync ? '/transform-async' : '/transform';

        } else {
            endpoint = '/transform-multi';
            const inputs = document.querySelectorAll('#multiExampleCard input[name="example_inputs"]');
            const outputs = document.querySelectorAll('#multiExampleCard input[name="example_outputs"]');

            let hasValidPair = false;
            inputs.forEach((input, i) => {
                if (input.files[0] && outputs[i]?.files[0]) {
                    formData.append('example_inputs', input.files[0]);
                    formData.append('example_outputs', outputs[i].files[0]);
                    filesToCheck.push(input.files[0], outputs[i].files[0]);
                    hasValidPair = true;
                }
            });

            if (!hasValidPair) {
                throw new Error('Please provide at least one complete example pair');
            }

            // Multi-example doesn't have async endpoint yet, use sync
            useAsync = false;
        }

        let result;
        if (useAsync) {
            result = await handleAsyncTransformation(formData, endpoint);
        } else {
            result = await handleSyncTransformation(formData, endpoint);
        }

        lastResult = result;

        // Display job ID
        document.getElementById('jobId').textContent = result.job_id;

        document.getElementById('analysis').textContent = result.transformation_analysis;
        document.getElementById('transformed').textContent = result.transformed_document;

        const metadata = result.metadata;
        document.getElementById('metadata').innerHTML = `
            <div class="metadata-item">Type: <span>${metadata.transformation_type}</span></div>
            <div class="metadata-item">Input Length: <span>${metadata.input_length.toLocaleString()}</span></div>
            <div class="metadata-item">Output Length: <span>${metadata.output_length.toLocaleString()}</span></div>
            ${metadata.chunks_processed > 1 ? `<div class="metadata-item">Chunks: <span>${metadata.chunks_processed}</span></div>` : ''}
            ${metadata.example_pairs_used ? `<div class="metadata-item">Examples Used: <span>${metadata.example_pairs_used}</span></div>` : ''}
            ${metadata.processing_mode ? `<div class="metadata-item">Mode: <span>${metadata.processing_mode}</span></div>` : ''}
        `;

        // Reset to result view and update comparison if diff data available
        setResultView('result');

        // Show/hide comparison toggle based on diff data availability
        const viewToggle = document.getElementById('viewToggle');
        if (result.diff_data && result.original_document) {
            viewToggle.style.display = 'flex';
        } else {
            viewToggle.style.display = 'none';
        }

        results.classList.add('visible');
        results.scrollIntoView({ behavior: 'smooth' });

        // Refresh history
        loadHistory();

    } catch (err) {
        error.textContent = err.message;
        error.classList.add('visible');
    } finally {
        submitBtn.disabled = false;
        loading.classList.remove('visible');
        hideProgress();
    }
});

function copyToClipboard(elementId) {
    const text = document.getElementById(elementId).textContent;
    navigator.clipboard.writeText(text).then(() => {
        // Show brief feedback
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
    const results = document.getElementById('results');
    const error = document.getElementById('error');

    try {
        const response = await fetch(`/jobs/${jobId}`);
        if (!response.ok) throw new Error('Failed to load job');

        const job = await response.json();
        lastResult = {
            job_id: job.job_id,
            transformation_analysis: job.transformation_analysis,
            transformed_document: job.transformed_document,
            original_document: job.original_document,
            diff_data: job.diff_data,
            metadata: job.metadata
        };

        document.getElementById('jobId').textContent = job.job_id;
        document.getElementById('analysis').textContent = job.transformation_analysis;
        document.getElementById('transformed').textContent = job.transformed_document;

        const metadata = job.metadata;
        document.getElementById('metadata').innerHTML = `
            <div class="metadata-item">Type: <span>${metadata.transformation_type}</span></div>
            <div class="metadata-item">Input Length: <span>${metadata.input_length.toLocaleString()}</span></div>
            <div class="metadata-item">Output Length: <span>${metadata.output_length.toLocaleString()}</span></div>
            ${metadata.chunks_processed > 1 ? `<div class="metadata-item">Chunks: <span>${metadata.chunks_processed}</span></div>` : ''}
            ${metadata.example_pairs_used ? `<div class="metadata-item">Examples Used: <span>${metadata.example_pairs_used}</span></div>` : ''}
            ${metadata.processing_mode ? `<div class="metadata-item">Mode: <span>${metadata.processing_mode}</span></div>` : ''}
        `;

        // Reset to result view and update comparison if diff data available
        setResultView('result');

        // Show/hide comparison toggle based on diff data availability
        const viewToggle = document.getElementById('viewToggle');
        if (job.diff_data && job.original_document) {
            viewToggle.style.display = 'flex';
        } else {
            viewToggle.style.display = 'none';
        }

        error.classList.remove('visible');
        results.classList.add('visible');
        results.scrollIntoView({ behavior: 'smooth' });

    } catch (err) {
        error.textContent = err.message;
        error.classList.add('visible');
    }
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    closeEventSource();
});

// Load history on page load
document.addEventListener('DOMContentLoaded', loadHistory);
