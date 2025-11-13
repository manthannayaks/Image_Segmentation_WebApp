const form = document.getElementById('upload-form');
const smoothImg = document.getElementById('smooth');
const randomImg = document.getElementById('random');
const previewImg = document.getElementById('preview');
const imageInput = document.getElementById('image');

imageInput.addEventListener('change', () => {
    const file = imageInput.files[0];
    if (file) {
        previewImg.src = URL.createObjectURL(file);
        const outputSection = document.getElementById('output-section');
        outputSection.style.display = 'block';
        outputSection.classList.add('show');
    } else {
        previewImg.src = "";
        const outputSection = document.getElementById('output-section');
        outputSection.style.display = 'none';
        outputSection.classList.remove('show');
    }
});

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const file = imageInput.files[0];
    if (!file) return alert("Please select an image first.");

    const button = form.querySelector('.segment-btn');
    const originalText = button.innerText;

    const statusMessage = document.getElementById('status-message');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    
    // Reset status message and progress
    statusMessage.style.display = 'none';
    statusMessage.innerText = '';
    progressBar.style.width = '0%';
    progressText.innerText = 'Processing 0%';
    statusMessage.classList.remove('show');
    
    // Disable the button and show loading text
    button.disabled = true;
    button.innerText = "Processing...";
    button.style.opacity = "0.6";

    const formData = new FormData();
    formData.append('image', file);

    const spinner = document.getElementById('loading-spinner');
    spinner.style.display = "block";

    const progressContainer = document.getElementById('progress-container');
    progressContainer.style.display = "block";
    let progress = 0;
    const interval = setInterval(() => {
        if (progress < 90) {
            progress += Math.floor(Math.random() * 10);
            if (progress > 90) progress = 90;
            progressBar.style.width = `${progress}%`;
            progressText.innerText = `Processing ${progress}%`;
        }
    }, 600);

    statusMessage.style.display = 'block';
    statusMessage.style.color = '#ff9800';
    statusMessage.innerText = 'Processing Image...';
    statusMessage.classList.add('show');

    try {
        const response = await fetch('http://127.0.0.1:5000/segment', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const data = await response.json();
            clearInterval(interval);
            progressBar.style.width = "100%";
            progressText.innerText = "Processing 100% - Done ✅";
            smoothImg.src = data.smooth + '?t=' + new Date().getTime();
            randomImg.src = data.random + '?t=' + new Date().getTime();

            statusMessage.style.display = 'block';
            statusMessage.style.color = '#4CAF50';
            statusMessage.innerText = 'Segmentation Completed Successfully ✅';
            statusMessage.classList.add('show');
            document.getElementById('output-section').classList.add('show');

            // ✅ Show toast notification
            const toast = document.createElement('div');
            toast.classList.add('toast');
            toast.innerText = "Segmentation Completed Successfully!";
            document.body.appendChild(toast);

            // Fade out toast after a few seconds
            setTimeout(() => {
                toast.style.opacity = '0';
                setTimeout(() => toast.remove(), 500);
            }, 2500);
        } else {
            // Removed alert here as per instructions
        }
    } catch (error) {
        statusMessage.style.display = 'block';
        statusMessage.style.color = '#f44336';
        statusMessage.innerText = 'Error: Segmentation Failed ❌';
        statusMessage.classList.add('show');

        // Removed alert here as per instructions
    }

    clearInterval(interval);
    progressContainer.style.display = "none";
    spinner.style.display = "none";
    
    // Re-enable the button
    button.disabled = false;
    button.innerText = originalText;
    button.style.opacity = "1";
});