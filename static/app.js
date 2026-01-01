// Smart Document Transformation - Frontend Application

let currentMode = 'single';
let pairCount = 1;
let lastResult = null;

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

    try {
        const formData = new FormData();
        const newDocument = document.querySelector('input[name="new_document"]').files[0];

        if (!newDocument) {
            throw new Error('Please select a document to transform');
        }

        formData.append('new_document', newDocument);

        let endpoint;

        if (currentMode === 'single') {
            endpoint = '/transform';
            const exampleInput = document.querySelector('#singleExampleCard input[name="example_input"]').files[0];
            const exampleOutput = document.querySelector('#singleExampleCard input[name="example_output"]').files[0];

            if (!exampleInput || !exampleOutput) {
                throw new Error('Please select both example input and output documents');
            }

            formData.append('example_input', exampleInput);
            formData.append('example_output', exampleOutput);
        } else {
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
        }

        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Transformation failed');
        }

        const result = await response.json();
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
        `;

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
    }
});

function copyToClipboard(elementId) {
    const text = document.getElementById(elementId).textContent;
    navigator.clipboard.writeText(text).then(() => {
        // Could add a toast notification here
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
        `;

        error.classList.remove('visible');
        results.classList.add('visible');
        results.scrollIntoView({ behavior: 'smooth' });

    } catch (err) {
        error.textContent = err.message;
        error.classList.add('visible');
    }
}

// Load history on page load
document.addEventListener('DOMContentLoaded', loadHistory);
