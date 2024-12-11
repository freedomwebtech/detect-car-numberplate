import os
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import numpy as np
import cvzone
import csv
from datetime import datetime, timedelta
import xlwings as xw

# Initialize PaddleOCR
ocr = PaddleOCR()
cap = cv2.VideoCapture('tc.mp4')
model = YOLO("best_float32.tflite")

# CSV file setup
csv_file = 'numberplates.csv'
excel_file = 'numberplates.xlsx'

# Create a new CSV file if it doesn't exist
def create_csv_file():
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Numberplate', 'Date', 'Time'])

# Create a new Excel file if it doesn't exist
def create_excel_file():
    if not os.path.exists(excel_file):
        wb = xw.Book()
        sheet = wb.sheets[0]
        sheet.range("A1").value = ["Numberplate", "Date", "Time"]
        wb.save(excel_file)
        wb.close()

create_csv_file()
create_excel_file()

# Read the class names for YOLO detection
with open("coco1.txt", "r") as f:
    class_names = f.read().splitlines()

# Function to perform OCR on an image array
def perform_ocr(image_array):
    if image_array is None:
        raise ValueError("Image is None")

    # Perform OCR on the image array
    results = ocr.ocr(image_array, rec=True)  # rec=True enables text recognition
    detected_text = []

    # Process OCR results
    if results[0] is not None:
        for result in results[0]:
            text = result[1][0]
            detected_text.append(text)

    # Join all detected texts into a single string
    return ''.join(detected_text)

# Function to get the current time in Indian Standard Time (IST)
def get_ist_time():
    utc_time = datetime.utcnow()  # Get the current UTC time
    ist_time = utc_time + timedelta(hours=5, minutes=30)  # Adjust to IST (UTC +5:30)
    return ist_time

# Function to update or append data to Excel
def update_excel(numberplate, date, time):
    try:
        # Open the Excel file
        wb = xw.Book(excel_file)
        sheet = wb.sheets[0]  # Access the first sheet

        # Find the last row with data
        last_row = sheet.used_range.last_cell.row

        # Append the new data to the next row
        sheet.range(f"A{last_row + 1}").value = [numberplate, date, time]

        # Save changes
        wb.save()
        print(f"Updated Excel with {numberplate}, {date}, {time}")
    except Exception as e:
        print(f"Error updating Excel: {e}")

# Mouse callback function to print mouse position
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Initialize video capture and YOLO model
count = 0
area = [(5, 180), (3, 249), (984, 237), (950, 168)]
counter = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))

    # Run YOLOv8 tracking on the frame
    results = model.track(frame, persist=True, imgsz=240)

    # Check if there are any boxes in the results
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
        class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
        track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
        confidences = results[0].boxes.conf.cpu().tolist()  # Confidence score

        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = class_names[class_id]
            x1, y1, x2, y2 = box
            cx = int(x1 + x2) // 2
            cy = int(y1 + y2) // 2

            result = cv2.pointPolygonTest(np.array(area, np.int32), ((cx, cy)), False)
            if result >= 0:
                if track_id not in counter:
                    counter.append(track_id)  # Only add if it's a new track ID

                    crop = frame[y1:y2, x1:x2]
                    crop = cv2.resize(crop, (120, 70))

                    # Perform OCR on the cropped image
                    text = perform_ocr(crop)
                    print(text)

                    # Clean the text for use as the filename
                    text_clean = text.replace('(', '').replace(')', '').replace(',', '').replace(']', '').replace('-', ' ').strip()

                    # If OCR detected text, save it to CSV and Excel
                    if text_clean:
                        # Get the current IST timestamp and date
                        current_ist_time = get_ist_time()
                        current_date = current_ist_time.date()  # Extract the date from IST timestamp

                        # Get the current IST time in HH:MM:SS format
                        current_time_str = current_ist_time.strftime("%H:%M:%S")  # Time in 24-hour format (IST)

                        # Open the CSV file and append the new data
                        with open(csv_file, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([text_clean, str(current_date), current_time_str])

                        print(f"Saved {text_clean} to CSV.")

                        # Update the Excel file
                        update_excel(text_clean, str(current_date), current_time_str)

                        # Save the cropped image with text as filename
                        image_filename = f"crops/{text_clean}.png"
                        if not os.path.exists("crops"):
                            os.makedirs("crops")  # Create a directory for crop images if it doesn't exist

                        cv2.imwrite(image_filename, crop)  # Save the cropped image
                        print(f"Saved cropped image as {image_filename}.")

    mycounter = len(counter)
    cvzone.putTextRect(frame, f'{mycounter}', (50, 60), 1, 1)
    cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 0), 2)
    cv2.imshow("RGB", frame)

    # Use cv2.waitKey(1) to allow real-time video frame processing
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Close video capture
cap.release()
cv2.destroyAllWindows()
