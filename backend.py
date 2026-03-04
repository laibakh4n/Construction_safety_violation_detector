"""
SafeWatch Backend — Construction Site Safety Monitor
====================================================
Requirements:
    pip install flask flask-cors ultralytics opencv-python pillow

Usage:
python backend.py --model best.pt
    python backend.py --model path/to/best.pt

Then open safety-monitor.html in your browser and set the backend URL to:
    http://localhost:5000/detect
"""

import argparse
import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)  # Allow browser requests from the HTML file

# ─── Class names matching data.yaml ───────────────────────────────────────────
CLASS_NAMES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask",
    "NO-Safety Vest", "Person", "Safety Cone",
    "Safety Vest", "machinery", "vehicle"
]

# Normalize to match HTML frontend keys (title-case)
NAME_MAP = {name.lower(): name for name in CLASS_NAMES}
NAME_MAP["machinery"] = "Machinery"
NAME_MAP["vehicle"] = "Vehicle"

model = None  # Loaded at startup


def load_model(model_path: str):
    global model
    print(f"[SafeWatch] Loading YOLO model from: {model_path}")
    model = YOLO(model_path)
    print(f"[SafeWatch] Model loaded. Classes: {model.names}")


def decode_image(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes to OpenCV BGR array."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/detect", methods=["POST"])
def detect():
    """
    Accepts a multipart form POST with field 'image' (JPEG/PNG bytes).
    Returns JSON:
    {
        "detections": { "Hardhat": 2, "NO-Hardhat": 1, ... },
        "boxes": [
            { "class": "NO-Hardhat", "conf": 0.87, "x": 120, "y": 80, "w": 90, "h": 130 }
        ],
        "risk_score": 42
    }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400

    file = request.files["image"]
    img_bytes = file.read()
    img = decode_image(img_bytes)

    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    # ─── Run YOLO inference ───────────────────────────────────────────────────
    results = model(img, verbose=False)[0]

    detections = {name: 0 for name in CLASS_NAMES}
    boxes_out = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        cls_name = results.names.get(cls_id, f"class_{cls_id}")

        # Normalize name
        normalized = NAME_MAP.get(cls_name.lower(), cls_name)

        if normalized in detections:
            detections[normalized] += 1

        # Bounding box in pixel coords (x, y, w, h)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        boxes_out.append({
            "class": normalized,
            "conf":  round(conf, 3),
            "x":     round(x1),
            "y":     round(y1),
            "w":     round(x2 - x1),
            "h":     round(y2 - y1),
        })

    # ─── Calculate risk score (mirrors frontend logic) ────────────────────────
    WEIGHTS = {
        "Hardhat":        -10,
        "Mask":            -5,
        "NO-Hardhat":      40,
        "NO-Mask":         20,
        "NO-Safety Vest":  30,
        "Person":           5,
        "Safety Cone":      0,
        "Safety Vest":     -5,
        "Machinery":       10,
        "Vehicle":          8,
    }
    raw_score = sum(detections.get(k, 0) * v for k, v in WEIGHTS.items())
    risk_score = max(0, min(100, round(raw_score)))

    return jsonify({
        "detections": detections,
        "boxes":      boxes_out,
        "risk_score": risk_score,
    })


@app.route("/detect_base64", methods=["POST"])
def detect_base64():
    """
    Alternative endpoint — accepts JSON body with base64-encoded image:
    { "image": "<base64 string>" }
    Useful for testing or non-multipart clients.
    """
    data = request.get_json(force=True)
    if not data or "image" not in data:
        return jsonify({"error": "No image in JSON body"}), 400

    img_bytes = base64.b64decode(data["image"])
    # Re-use the same logic by injecting into a fake request
    img = decode_image(img_bytes)
    if img is None:
        return jsonify({"error": "Could not decode base64 image"}), 400

    results = model(img, verbose=False)[0]
    detections = {name: 0 for name in CLASS_NAMES}
    boxes_out  = []

    for box in results.boxes:
        cls_id   = int(box.cls[0])
        conf     = float(box.conf[0])
        cls_name = results.names.get(cls_id, f"class_{cls_id}")
        normalized = NAME_MAP.get(cls_name.lower(), cls_name)
        if normalized in detections:
            detections[normalized] += 1
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        boxes_out.append({"class": normalized, "conf": round(conf,3),
                          "x": round(x1), "y": round(y1),
                          "w": round(x2-x1), "h": round(y2-y1)})

    WEIGHTS = {"Hardhat":-10,"Mask":-5,"NO-Hardhat":40,"NO-Mask":20,
               "NO-Safety Vest":30,"Person":5,"Safety Cone":0,
               "Safety Vest":-5,"Machinery":10,"Vehicle":8}
    risk_score = max(0, min(100, round(sum(detections.get(k,0)*v for k,v in WEIGHTS.items()))))

    return jsonify({"detections": detections, "boxes": boxes_out, "risk_score": risk_score})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SafeWatch Detection Backend")
    parser.add_argument("--model", type=str, default="best.pt",
                        help="Path to trained YOLO weights (.pt file)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Confidence threshold for detections")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        print("  Please provide your trained weights file, e.g.:")
        print("  python backend.py --model runs/train/weights/best.pt")
        exit(1)

    load_model(str(model_path))
    print(f"\n[SafeWatch] Server starting at http://{args.host}:{args.port}")
    print(f"[SafeWatch] Detection endpoint: POST http://localhost:{args.port}/detect")
    print(f"[SafeWatch] Open safety-monitor.html in your browser\n")

    app.run(host=args.host, port=args.port, debug=False)
