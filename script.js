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

    // Reset UI
    statusMessage.style.display = 'block';
    statusMessage.style.color = '#ff9800';
    statusMessage.innerText = 'Processing Image...';
    progressBar.style.width = '0%';
    progressText.innerText = 'Processing 0%';
    button.disabled = true;
    button.innerText = "Processing...";
    button.style.opacity = "0.6";

    const spinner = document.getElementById('loading-spinner');
    spinner.style.display = "block";

    const progressContainer = document.getElementById('progress-container');
    progressContainer.style.display = "block";

    // Fake progress animation
    let progress = 0;
    const interval = setInterval(() => {
        if (progress < 95) {
            progress += Math.floor(Math.random() * 10);
            if (progress > 95) progress = 95;
            progressBar.style.width = `${progress}%`;
            progressText.innerText = `Processing ${progress}%`;
        }
    }, 500);

    try {
        // Wait for OpenCV
        if (typeof cv === 'undefined' || !cv.getBuildInformation) {
            statusMessage.innerText = 'Waiting for OpenCV to load...';
            await new Promise(resolve => {
                const check = setInterval(() => {
                    if (cv && cv.getBuildInformation) {
                        clearInterval(check);
                        resolve();
                    }
                }, 50);
            });
        }

        // Load image from file into cv.Mat
        const img = new Image();
        img.src = URL.createObjectURL(file);

        await new Promise(res => img.onload = res);

        const canvas = document.createElement("canvas");
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0);

        let mat = cv.imread(canvas);

        // ⭐ Call your segmentation function (from segment.js)
        const result = await segmentImage(mat, {
            minSize: 50,
            connectivity: 4,
            meanThreshold: 20
        });

        // Show results
        smoothImg.src = result.meanCanvas.toDataURL();
        randomImg.src = result.randomCanvas.toDataURL();

        clearInterval(interval);
        progressBar.style.width = "100%";
        progressText.innerText = "Processing 100% - Done ✅";

        statusMessage.style.color = '#4CAF50';
        statusMessage.innerText = 'Segmentation Completed Successfully ✅';

        // Cleanup
        mat.delete();

    } catch (error) {
        console.error(error);
        statusMessage.style.color = '#f44336';
        statusMessage.innerText = 'Error: Segmentation Failed ❌';
    }

    clearInterval(interval);
    spinner.style.display = "none";
    progressContainer.style.display = "none";

    button.disabled = false;
    button.innerText = originalText;
    button.style.opacity = "1";
});