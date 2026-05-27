import cv2
import numpy as np
import time
import os
import subprocess
from datetime import datetime
from picamera2 import Picamera2

# =========================
# Path settings
# =========================
BASE_DIR = os.path.expanduser("~/plate-detection-event")
MODEL_PATH = os.path.join(BASE_DIR, "models", "license_plate_yolov5s.onnx")
CAPTURE_DIR = os.path.join(BASE_DIR, "captures")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CROP_DIR = os.path.join(OUTPUT_DIR, "crops")

os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)

# =========================
# Model settings
# =========================
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
CONF_THRESHOLD = 0.35
NMS_THRESHOLD = 0.45

# =========================
# Helper functions
# =========================
def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize image while keeping aspect ratio.
    This is commonly used for YOLO input.
    """
    shape = image.shape[:2]  # height, width
    h, w = shape

    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )

    return padded, r, dw, dh


def run_tesseract_ocr(crop_path):
    """
    Run Tesseract OCR on cropped plate image.
    This is optional. The main assignment result is detection.
    """
    try:
        result = subprocess.run(
            ["tesseract", crop_path, "stdout", "--psm", "7"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        text = result.stdout.strip()
        return text
    except Exception as e:
        return f"OCR error: {e}"


def detect_plate(image, net):
    """
    Run ONNX YOLO model and return detected boxes.
    """
    original = image.copy()
    original_h, original_w = original.shape[:2]

    input_img, ratio, dw, dh = letterbox(original, (INPUT_HEIGHT, INPUT_WIDTH))

    blob = cv2.dnn.blobFromImage(
        input_img,
        scalefactor=1 / 255.0,
        size=(INPUT_WIDTH, INPUT_HEIGHT),
        swapRB=True,
        crop=False
    )

    net.setInput(blob)
    outputs = net.forward()

    # Output shape can be (1, N, 6) or (N, 6)
    outputs = np.squeeze(outputs)

    boxes = []
    confidences = []

    if len(outputs.shape) == 1:
        outputs = np.expand_dims(outputs, axis=0)

    for det in outputs:
        # Expected format: x, y, w, h, confidence, class_score...
        if len(det) < 5:
            continue

        x, y, w, h = det[0], det[1], det[2], det[3]
        confidence = det[4]

        # Some YOLO exports include class scores after index 5
        if len(det) > 5:
            class_score = np.max(det[5:])
            confidence = confidence * class_score

        if confidence < CONF_THRESHOLD:
            continue

        # Convert YOLO center format to corner format
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        # Remove letterbox padding and scale back to original image
        x1 = (x1 - dw) / ratio
        y1 = (y1 - dh) / ratio
        x2 = (x2 - dw) / ratio
        y2 = (y2 - dh) / ratio

        x1 = int(max(0, min(original_w - 1, x1)))
        y1 = int(max(0, min(original_h - 1, y1)))
        x2 = int(max(0, min(original_w - 1, x2)))
        y2 = int(max(0, min(original_h - 1, y2)))

        box_w = x2 - x1
        box_h = y2 - y1

        if box_w <= 0 or box_h <= 0:
            continue

        boxes.append([x1, y1, box_w, box_h])
        confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        CONF_THRESHOLD,
        NMS_THRESHOLD
    )

    results = []

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            conf = confidences[i]
            results.append((x, y, w, h, conf))

    return results


# =========================
# Main
# =========================
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        return

    print("Loading ONNX model...")
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)

    print("Starting Raspberry Pi Camera...")
    picam2 = Picamera2()

    config = picam2.create_still_configuration(
        main={"size": (1280, 720)}
    )
    picam2.configure(config)
    picam2.start()

    time.sleep(2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    capture_path = os.path.join(CAPTURE_DIR, f"capture_{timestamp}.jpg")
    output_path = os.path.join(OUTPUT_DIR, f"result_{timestamp}.jpg")

    print("Capturing image...")
    frame = picam2.capture_array()
    picam2.stop()

    # Picamera2 usually returns RGB, OpenCV uses BGR
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imwrite(capture_path, frame_bgr)
    print(f"Captured image saved: {capture_path}")

    print("Running license plate detection...")
    detections = detect_plate(frame_bgr, net)

    print(f"Detected plates: {len(detections)}")

    for idx, (x, y, w, h, conf) in enumerate(detections):
        x2 = x + w
        y2 = y + h

        # Draw bounding box
        cv2.rectangle(frame_bgr, (x, y), (x2, y2), (0, 255, 0), 2)

        label = f"plate {conf:.2f}"
        cv2.putText(
            frame_bgr,
            label,
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Crop detected plate
        crop = frame_bgr[y:y2, x:x2]
        crop_path = os.path.join(CROP_DIR, f"plate_{timestamp}_{idx}.jpg")
        cv2.imwrite(crop_path, crop)

        print(f"Crop saved: {crop_path}")

        # Optional OCR
        ocr_text = run_tesseract_ocr(crop_path)
        print(f"OCR result for plate {idx}: {ocr_text}")

    cv2.imwrite(output_path, frame_bgr)
    print(f"Result image saved: {output_path}")

    print("Done.")


if __name__ == "__main__":
    main()
