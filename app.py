from flask import Flask, request, jsonify, send_from_directory
import os
import cv2 as cv
import numpy as np
from main import segment_graph, random_color_segments  # from main.py logic

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/segment', methods=['POST'])
def segment_image():
    file = request.files['image']
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # Save uploaded image
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    print("Saved path:", path)
    print("File exists:", os.path.exists(path))
    # Read image
    img = cv.imread(path)
    if img is None:
        return jsonify({"error": "Invalid image file"}), 400

    # Segment
    labels, vis = segment_graph(img,min_size=50,connectivity=8,mean_threshold=20)
    rand_vis = random_color_segments(labels)

    # Save results
    smooth_path = os.path.join(OUTPUT_FOLDER, "smooth_" + file.filename)
    random_path = os.path.join(OUTPUT_FOLDER, "random_" + file.filename)
    cv.imwrite(smooth_path, vis)
    cv.imwrite(random_path, rand_vis)

    return jsonify({
    "smooth": f"http://127.0.0.1:5000/outputs/smooth_{file.filename}",
    "random": f"http://127.0.0.1:5000/outputs/random_{file.filename}"
})

@app.route('/uploads/<filename>')
def serve_uploaded(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/outputs/<filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/')
def home():
    return send_from_directory(os.path.join(BASE_DIR, 'Frontend'), 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(os.path.join(BASE_DIR, 'Frontend'), path)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)