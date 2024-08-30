import requests
import time
import torch
from ultralytics import YOLO
from math import radians, cos, sin, sqrt, atan2
import cv2
import os

# Hardcoded Latitude and Longitude
latitude = 51.44448483224641
longitude = -0.4036596372128636

# Google Maps API key
api_key = ""

# YOLOv8 model
model = YOLO("C:/Users/yadav/Desktop/knifedetection/best.pt")

# Function to calculate the distance between two lat/lng points using the Haversine formula
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    return distance

# Function to find the nearest police station
def find_nearest_police_station():
    endpoint_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{latitude},{longitude}",
        "radius": 5000,  # Search within 5km radius
        "type": "police",  # Search for police stations
        "key": api_key
    }

    response = requests.get(endpoint_url, params=params)
    results = response.json()

    if results["status"] == "OK" and results["results"]:
        nearest_place = None
        min_distance = float('inf')

        for place in results["results"]:
            place_location = place["geometry"]["location"]
            distance = calculate_distance(latitude, longitude, place_location["lat"], place_location["lng"])

            if distance < min_distance:
                min_distance = distance
                nearest_place = place

        if nearest_place:
            nearest_name = nearest_place["name"]
            nearest_address = nearest_place.get("vicinity", "No address available")

            return nearest_name, nearest_address

    return None, None

# Function to save the image with detections and the cooldown label
def save_image_with_detections(image, results, save_path, label_text=""):
    annotated_frame = results[0].plot()  # Use the plot method on the first result in the list
    if label_text:
        cv2.putText(annotated_frame, label_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(save_path, annotated_frame)

# Main detection loop
cooldown_time = 60  # Cooldown time in seconds
last_detection_time = 0
in_cooldown = False  # Track if cooldown is active

# Create a folder for saving detection images
save_folder = "C:/Users/yadav/Desktop/knifedetection/images"
os.makedirs(save_folder, exist_ok=True)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    if in_cooldown:
        # Display cooldown message on video feed
        cv2.putText(frame, "Knife detected, in process of cooldown", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        # Check if cooldown period is over
        if current_time - last_detection_time > cooldown_time:
            in_cooldown = False
        else:
            # Continue to display video feed without detection during cooldown
            cv2.imshow("Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
    else:
        # Perform detection
        results = model(frame)
        
        # Extract detected labels
        detected_labels = []
        if results[0].boxes is not None:
            detected_labels = [model.names[int(box.cls[0])] for box in results[0].boxes]

        # Check if 'Knife_Deploy' or 'Knife_Weapon' is detected
        if "Knife_Deploy" in detected_labels or "Knife_Weapon" in detected_labels:
            # Save the image with bounding boxes
            save_image_path = os.path.join(save_folder, f"knife_detection_{int(current_time)}.jpg")
            save_image_with_detections(frame, results, save_image_path)
            print(f"Knife detected. Image saved to {save_image_path}")

            # Find the nearest police station
            nearest_name, nearest_address = find_nearest_police_station()
            if nearest_name and nearest_address:
                print(f"Nearest Police Station: {nearest_name}, Address: {nearest_address}")
            else:
                print("No police stations found nearby.")

            # Update the last detection time and start cooldown
            last_detection_time = current_time
            in_cooldown = True

    # Show the video feed
    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

