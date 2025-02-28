from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from ultralytics import YOLO
from io import BytesIO
import base64
import pytesseract

# Load the trained YOLO model
model = YOLO("best.pt")
# results = model("generate_id1740757269.jpg")
# # results[0].show()

# for result in results:
#     result.show()  # Show detection with bounding boxes

#     # Extract detected information
#     detections = []
#     for box in result.boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
#         conf = float(box.conf[0])  # Confidence score
#         cls = int(box.cls[0])  # Class index
#         label = model.names[cls]  # Class label (e.g., 'Name', 'ID_Number')

#         detections.append({
#             "label": label,
#             "confidence": round(conf, 2),
#             "bbox": [x1, y1, x2, y2]
#         })

#     print(detections)  # Print detected values in a structured format
# Initialize FastAPI
app = FastAPI()

# # Function to read image
# async def read_image(file):
#     contents = await file.read()
#     np_array = np.frombuffer(contents, np.uint8)
#     img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
#     return img

# def extract_text_from_region(img, bbox):
#     x1, y1, x2, y2 = bbox
#     roi = img[y1:y2, x1:x2]  # Crop the detected region
#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
#     text = pytesseract.image_to_string(gray, lang="eng")  # Extract text
#     return text.strip()

# # API Endpoint to detect national ID details
# @app.post("/detect")
# async def detect_id(file: UploadFile = File(...)):
#     # Read image
#     img = await read_image(file)

#     # Run YOLO detection
#     results = model(img)

#     # Extract detections
#     detections = []
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
#             conf = float(box.conf[0])  # Confidence score
#             cls = int(box.cls[0])  # Class index
#             label = model.names[cls]  # Class label (e.g., 'Name', 'ID_Number')

#             detections.append({
#                 "label": label,
#                 "confidence": round(conf, 2),
#                 "bbox": [x1, y1, x2, y2]
#             })

#     return {"detections": detections}

# Function to read image from request
async def read_image(file: UploadFile):
    contents = await file.read()  # Read file asynchronously
    np_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return img

# Function to extract text using OCR
# def extract_text_from_region(img, bbox):
#     x1, y1, x2, y2 = bbox
#     roi = img[y1:y2, x1:x2]  # Crop the detected region
#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
#     text = pytesseract.image_to_string(gray, lang="eng")  # Extract text
#     return text.strip()
# def preprocess_image(roi):
#     """ Preprocess image for better OCR accuracy """
#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    
#     # Apply Adaptive Thresholding
#     gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
#     # Denoise image
#     gray = cv2.fastNlMeansDenoising(gray, h=30)

#     return gray

# def extract_text_from_region(img, bbox):
#     """ Extract text from detected region using Tesseract OCR """
#     x1, y1, x2, y2 = bbox
#     roi = img[y1:y2, x1:x2]  # Crop detected region

#     # Preprocess the image
#     processed_img = preprocess_image(roi)

#     # Use Persian (Farsi) and English OCR
#     # text = pytesseract.image_to_string(processed_img, lang="fas+eng", config="--psm 6")
#     text = pytesseract.image_to_string(
#     processed_img, lang="fas+eng", config="--psm 7 -c tessedit_char_whitelist=0123456789/ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي"
# )

#     return text.strip()
def preprocess_image(roi):
    """ Preprocess image for better OCR accuracy """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    
    # Increase contrast
    gray = cv2.equalizeHist(gray)

    # Apply binary thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Denoise image
    denoised = cv2.fastNlMeansDenoising(thresh, h=20)

    # Morphological operations (dilation + erosion)
    kernel = np.ones((2,2), np.uint8)
    processed_img = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)

    return processed_img

def extract_text_from_region(img, bbox):
    """ Extract text from detected region using Tesseract OCR """
    x1, y1, x2, y2 = bbox
    roi = img[y1:y2, x1:x2]  # Crop detected region

    # Preprocess the image
    processed_img = preprocess_image(roi)

    # Use Persian (Farsi) and English OCR
    text = pytesseract.image_to_string(processed_img, lang="fas+eng", config="--psm 6")
    # text = pytesseract.image_to_string(
    # processed_img, lang="fas+eng", config="--psm 7 -c tessedit_char_whitelist=0123456789/ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي"
    # )

    return text.strip()
@app.post("/detect/")
async def detect_id(file: UploadFile = File(...)):
    img = await read_image(file)  # Read uploaded image

    # Run YOLO model
    results = model(img)

    # Extract detected data with OCR
    detected_fields = {}
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            conf = round(float(box.conf[0]), 2)  # Confidence
            cls = int(box.cls[0])  # Class index
            label = model.names[cls]  # Class name (e.g., "national id")

            # Extract text from detected region
            text = extract_text_from_region(img, (x1, y1, x2, y2))

            # Store detected values
            detected_fields[label] = {
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "text": text  # Extracted text
            }

    return {"detected_fields": detected_fields}