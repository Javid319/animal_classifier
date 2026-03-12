// ========================================
// Animal Classifier - JavaScript
// ========================================

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const uploadSection = document.getElementById('uploadSection');
const previewSection = document.getElementById('previewSection');
const loadingSection = document.getElementById('loadingSection');
const resultSection = document.getElementById('resultSection');
const imagePreview = document.getElementById('imagePreview');
const resultValue = document.getElementById('resultValue');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceFill = document.getElementById('confidenceFill');
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const tryAnotherBtn = document.getElementById('tryAnotherBtn');

// ========================================
// Event Listeners
// ========================================

// File input change
imageInput.addEventListener('change', handleFileSelect);

// Drag and drop events
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// Click on upload area to trigger file input
uploadArea.addEventListener('click', () => {
    imageInput.click();
});

// Button click handlers
clearBtn.addEventListener('click', clearPreview);
predictBtn.addEventListener('click', predictImage);
tryAnotherBtn.addEventListener('click', resetUI);

// ========================================
// Functions
// ========================================

function handleFileSelect(e) {
    const file = e.target.files[0];
    handleFile(file);
}

function handleFile(file) {
    if (!file) return;
    
    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file!');
        return;
    }
    
    // Read and preview the image
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        showPreview();
    };
    reader.readAsDataURL(file);
}

function showPreview() {
    uploadSection.style.display = 'none';
    previewSection.style.display = 'block';
    resultSection.style.display = 'none';
    loadingSection.style.display = 'none';
}

function clearPreview() {
    imageInput.value = '';
    imagePreview.src = '';
    resetUI();
}

function resetUI() {
    uploadSection.style.display = 'block';
    previewSection.style.display = 'none';
    resultSection.style.display = 'none';
    loadingSection.style.display = 'none';
}

async function predictImage() {
    const file = imageInput.files[0];
    if (!file) {
        alert('Please select an image first!');
        return;
    }
    
    // Show loading state
    previewSection.style.display = 'none';
    loadingSection.style.display = 'block';
    resultSection.style.display = 'none';
    
    // Create form data
    const formData = new FormData();
    formData.append('image', file);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const data = await response.json();
        showResult(data.prediction, data.confidence || 0.95);
        
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to make prediction. Please try again.');
        resetUI();
    }
}

function showResult(prediction, confidence) {
    loadingSection.style.display = 'none';
    resultSection.style.display = 'block';
    
    // Add fade-in animation
    resultSection.classList.add('fade-in');
    
    // Update result
    resultValue.textContent = prediction;
    
    // Update confidence
    const confidencePercent = Math.round(confidence * 100);
    confidenceValue.textContent = confidencePercent + '%';
    confidenceFill.style.width = confidencePercent + '%';
    
    // Add icon based on animal
    const resultIcon = document.getElementById('resultIcon');
    const iconMap = {
        'Buffalo': 'bi-hourglass-split',
        'Elephant': 'bi-tree',
        'Rhino': 'bi-shield-exclamation',
        'Zebra': 'bi-columns-gap'
    };
    
    if (iconMap[prediction]) {
        resultIcon.innerHTML = `<i class="bi ${iconMap[prediction]}"></i>`;
    }
}

