
from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import io
import pytesseract

app = FastAPI()

# Load YOLO model
model = YOLO("best.pt")  # Ensure best.pt is in the same directory

# Class labels (ensure these match your YOLO training classes)
LABELS = ["National ID", "Name", "Surname", "Date of Birth", "Father's Name", "Expiration Date"]

def preprocess_image(image_bytes):
    """ Convert image to OpenCV format. """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(image)

@app.post("/detect_boxes")
async def detect_bounding_boxes(file: UploadFile = File(...)):
    """Detect bounding boxes without running OCR."""
    image_bytes = await file.read()
    image = preprocess_image(image_bytes)

    # Run YOLO detection
    results = model(image)

    detected_boxes = []
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, class_id = map(float, box)  # Convert to float
            detected_boxes.append({
                "label": LABELS[int(class_id)],
                "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "confidence": conf
            })

    return {"bounding_boxes": detected_boxes}

@app.post("/extract")
async def extract_national_id(file: UploadFile = File(...)):
    """Detect bounding boxes and extract text using OCR."""
    image_bytes = await file.read()
    image = preprocess_image(image_bytes)

    # Run YOLO detection
    results = model(image)
    # print("===>1",results)
    extracted_data = {}
    for result in results:
        # print("===>2",result)
        for box in result.boxes.data:
            # print("===>3",result)
            x1, y1, x2, y2, conf, class_id = map(int, box)  # Convert to integer
            roi = image[y1:y2, x1:x2]  # Crop detected region

            # Convert to grayscale before OCR
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray, lang="fas", config="--psm 6").strip()

            extracted_data[LABELS[class_id]] = text

    return {"Extracted Data": extracted_data}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
