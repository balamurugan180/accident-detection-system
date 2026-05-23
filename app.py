import os
import cv2
import uuid
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from detection import AccidentDetectionModel

# -------------------------------
# Flask Configuration
# -------------------------------

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "mp4", "avi", "mov"}

current_video = None

font = cv2.FONT_HERSHEY_SIMPLEX

# -------------------------------
# Load Models
# -------------------------------

accident_model = AccidentDetectionModel("model.json", "model_weights.h5")
yolo_model = YOLO("yolov8n.pt")

VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "bicycle"}
PEDESTRIAN_CLASS = "person"


# -------------------------------
# Helper Functions
# -------------------------------

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_severity(accident, confidence, vehicles, pedestrians):

    if not accident:
        return "Low"

    if confidence > 95 and (vehicles >= 3 or pedestrians >= 2):
        return "High"

    if confidence > 85:
        return "Moderate"

    return "Low"


def get_recommendations(severity):

    if severity == "High":
        return [
            "Dispatch emergency services immediately.",
            "Alert traffic control authorities.",
            "Divert nearby vehicles.",
            "Notify medical response teams."
        ]

    if severity == "Moderate":
        return [
            "Monitor situation closely.",
            "Alert nearby traffic patrol.",
            "Prepare emergency response."
        ]

    return [
        "Continue monitoring the traffic scene."
    ]


def generate_summary(accident, confidence, vehicles, pedestrians, severity):

    if accident:

        return f"""
The AI system detected a road accident with {confidence}% confidence.
Detected {vehicles} vehicle(s) and {pedestrians} pedestrian(s) in the scene.
Based on traffic density the severity level is {severity}.
"""

    return "No accident patterns detected."


# -------------------------------
# Object Detection
# -------------------------------

def detect_objects(frame):

    vehicle_count = 0
    pedestrian_count = 0

    results = yolo_model(frame, verbose=False)

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            cls_id = int(box.cls[0])
            label = yolo_model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label in VEHICLE_CLASSES:

                vehicle_count += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255,140,0), 2)
                cv2.putText(frame, label, (x1, y1-10), font, 0.6, (255,140,0), 2)

            elif label == PEDESTRIAN_CLASS:

                pedestrian_count += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,200,0), 2)
                cv2.putText(frame, label, (x1, y1-10), font, 0.6, (0,200,0), 2)

    return frame, vehicle_count, pedestrian_count


# -------------------------------
# Accident Detection
# -------------------------------

def detect_accident(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    roi = cv2.resize(rgb, (250,250))

    pred, prob = accident_model.predict_accident(roi[np.newaxis,:,:])

    if pred == "Accident":

        confidence = round(float(prob[0][0]) * 100, 2)

        return True, confidence

    return False, 0


# -------------------------------
# Frame Processing
# -------------------------------

def process_frame(frame):

    annotated = frame.copy()

    annotated, vehicles, pedestrians = detect_objects(annotated)

    accident, confidence = detect_accident(frame)

    severity = get_severity(accident, confidence, vehicles, pedestrians)

    summary = generate_summary(accident, confidence, vehicles, pedestrians, severity)

    if accident:

        cv2.putText(
            annotated,
            f"ACCIDENT DETECTED {confidence}%",
            (20,40),
            font,
            1,
            (0,0,255),
            3
        )

    return {
        "frame": annotated,
        "accident": accident,
        "confidence": confidence,
        "vehicle_count": vehicles,
        "pedestrian_count": pedestrians,
        "severity": severity,
        "summary": summary,
        "recommendations": get_recommendations(severity)
    }


# -------------------------------
# Video Streaming
# -------------------------------

def generate_stream():

    global current_video

    cap = cv2.VideoCapture(current_video)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        result = process_frame(frame)

        frame = result["frame"]

        ret, buffer = cv2.imencode(".jpg", frame)

        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame_bytes + b'\r\n'
        )

    cap.release()


# -------------------------------
# Routes
# -------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# -------------------------------
# Video Detection Page
# -------------------------------

@app.route("/video_detection")
def video_detection():
    return render_template("video_detection.html")


@app.route("/video_feed")
def video_feed():

    if current_video is None:
        return ""

    return Response(
        generate_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# -------------------------------
# Upload Video
# -------------------------------

@app.route("/upload_video", methods=["POST"])
def upload_video():

    global current_video

    file = request.files["video"]

    filename = secure_filename(file.filename)

    unique_name = f"{uuid.uuid4().hex}_{filename}"

    path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)

    file.save(path)

    current_video = path

    cap = cv2.VideoCapture(path)

    accident_result = None

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        result = process_frame(frame)

        if result["accident"]:

            accident_result = result
            break

    cap.release()

    if accident_result is None:

        return render_template(
            "video_detection.html",
            uploaded_video=f"uploads/{unique_name}",
            no_accident=True
        )

    return render_template(
        "video_detection.html",
        uploaded_video=f"uploads/{unique_name}",
        result_probability=accident_result["confidence"],
        vehicle_count=accident_result["vehicle_count"],
        pedestrian_count=accident_result["pedestrian_count"],
        severity=accident_result["severity"],
        summary_report=accident_result["summary"],
        recommendations=accident_result["recommendations"],
        accident_detected=True
    )


# -------------------------------
# Image Upload
# -------------------------------

@app.route("/image_upload", methods=["GET","POST"])
def image_upload():

    if request.method == "POST":

        file = request.files["image"]

        filename = secure_filename(file.filename)

        unique_name = f"{uuid.uuid4().hex}_{filename}"

        path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)

        file.save(path)

        img = cv2.imread(path)

        result = process_frame(img)

        result_name = f"result_{unique_name}"

        result_path = os.path.join(app.config["RESULT_FOLDER"], result_name)

        cv2.imwrite(result_path, result["frame"])

        return render_template(
            "results.html",
            uploaded_image=f"uploads/{unique_name}",
            result_image=f"results/{result_name}",
            result_probability=result["confidence"],
            vehicle_count=result["vehicle_count"],
            pedestrian_count=result["pedestrian_count"],
            severity=result["severity"],
            summary_report=result["summary"],
            recommendations=result["recommendations"]
        )

    return render_template("image_upload.html")


# -------------------------------
# Run Flask
# -------------------------------

if __name__ == "__main__":
    app.run(debug=True)