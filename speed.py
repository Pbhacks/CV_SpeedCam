import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *
import time

# Load YOLO model
model = YOLO('yolov8n.pt')

# Open video file or live camera feed
cap = cv2.VideoCapture('veh2.mp4')  # Use 0 for live camera, replace with 'veh2.mp4' for video file

# Read class names from coco.txt
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Initialize tracker and vehicle tracking variables
tracker = Tracker()
vehicle_speeds = {}
last_frame_time = {}
total_vehicle_count = 0
frame_count = 0

# Distance between frames in meters for speed calculation (assuming consistent distance)
distance_between_frames = 5
# Process every Nth frame
# Process every Nth frame
process_frame_interval = 5

fps = cap.get(cv2.CAP_PROP_FPS)
time_per_frame = 5 / fps  # Time elapsed between two consecutive frames

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    
    # Resize the frame for easier processing
    frame = cv2.resize(frame, (1020, 500))

    # Run YOLO object detection
    results = model.predict(frame)
    detections = results[0].boxes.data
    px = pd.DataFrame(detections).astype("float")

    # List to store detected bounding boxes for cars
    car_list = []
    
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        class_id = int(row[5])
        class_name = class_list[class_id]
        
        # Only track cars
        if 'car' in class_name:
            car_list.append([x1, y1, x2, y2])

    # Update tracker with detected cars in the current frame
    bbox_id = tracker.update(car_list)

    current_time = time.time()
    
    for bbox in bbox_id:
        x3, y3, x4, y4, vehicle_id = bbox
        cx = int((x3 + x4) // 2)  # Center x
        cy = int((y3 + y4) // 2)  # Center y

        # Draw rectangle around the detected vehicle
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
        
        # If this is a new vehicle, increase the total count
        if vehicle_id not in vehicle_speeds:
            total_vehicle_count += 1
            vehicle_speeds[vehicle_id] = 0  # Initialize speed to zero
            last_frame_time[vehicle_id] = current_time  # Store the current time

        # Calculate speed
        prev_time = last_frame_time[vehicle_id]
        time_elapsed = current_time - prev_time
        if time_elapsed > 0:
            speed_mps = distance_between_frames / time_elapsed  # Speed in meters/second
            speed_kmph = speed_mps * 3.6  # Convert to km/h
            
            # Update vehicle speed and last time
            vehicle_speeds[vehicle_id] = speed_kmph
            last_frame_time[vehicle_id] = current_time
         
            # Display speed on frame
            cv2.putText(frame, f"ID: {vehicle_id} {int(speed_kmph)} km/h", (x3, y3-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)
        
        # Display vehicle count on frame
        cv2.putText(frame, f"Total Vehicles: {total_vehicle_count}", (60, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    # Show the frame with detection, speed, and vehicle count
    cv2.imshow("Vehicle Detection", frame)

    # Exit when the 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
