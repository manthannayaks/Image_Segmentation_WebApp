from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2 as cv
import numpy as np
import os
from hi import segment_graph

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/segment', methods=['POST'])
def segment():
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    print(f"✅ Uploaded: {filepath}")

    img_bgr = cv.imread(filepath)
    if img_bgr is None:
        return "Failed to read image", 500

    # Run segmentation
    labels, vis = segment_graph(img_bgr, k=200.0, min_size=50)

    # Save smooth segmented image
    smooth_path = os.path.join(OUTPUT_FOLDER, "segmented_smooth.png")
    cv.imwrite(smooth_path, vis)

    # Create and save random color map
    unique_labels = np.unique(labels)
    color_map = np.random.randint(0, 255, (len(unique_labels), 3), dtype=np.uint8)
    random_colored = color_map[labels]
    random_path = os.path.join(OUTPUT_FOLDER, "segmented_random.png")
    cv.imwrite(random_path, random_colored)

    print("✅ Saved both images!")

    # Return both paths
    return jsonify({
        "smooth": "/outputs/segmented_smooth.png",
        "random": "/outputs/segmented_random.png"
    })

# Allow Flask to serve files from outputs/
@app.route('/outputs/<path:filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)