import cv2
import pandas as pd
from ultralytics import YOLO
import time
import tkinter as tk
from tkinter import messagebox

def start(source_path, od_function):

    cap = cv2.VideoCapture(source_path)
    model = YOLO('YOLOv8/config/truck.pt')
    class_file = open("YOLOv8/config/truck_class.txt", "r").read()    
    class_list = class_file.split("\n")
    # print(class_list)

    count = 0

    # Set the line position
    line_position = 420  # Adjust this value based on your video dimensions

    total_fps = 0.0
    num_frames = 0
    max_confidence = 0.0
    max_confidence_od = 0.0

    start_time = time.time()  # Initialize start_time
    start_processing_time = time.time()

    object_detected = False  # Flag to track if object has been detected

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        count += 1
        if count % 3 != 0:
            continue

        frame = cv2.resize(frame, (640, 640))
        results = model.predict(frame)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")
        height, width, channels = frame.shape

        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            c = class_list[d]
            confidence = row[4]  # Confidence score

            w = int(row[2] * width)
            h = int(row[3] * height)

            # Check if the low part of the bounding box is near the line
            if (line_position - 10) < y2 < line_position and not object_detected:
                object_detected = True  # Set the flag to True                                    
                # Export the cropped image only if object has not been detected yet                    
                object_image = frame[y1:y2, x1:x2]            
                object_image_resized = cv2.resize(object_image, (640, 640))

                # Call od_detection function with the resized cropped object as parameter
                confidence_od = od_function(object_image_resized)
                if confidence_od > max_confidence_od:
                    max_confidence_od = confidence_od
                    
            if y2 > (line_position + 6):
                object_detected = False

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f'{c} ({confidence:.2f})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            if confidence > max_confidence:
                max_confidence = confidence

        # Draw the line
        cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 255, 0), 2)

        # Calculate FPS
        fps = 1.0 / (time.time() - start_time)
        start_time = time.time()

        # Accumulate total FPS
        total_fps += fps
        num_frames += 1

        # Calculate average FPS
        avg_fps = total_fps / num_frames

        # Overlay FPS and average confidence score on frame
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("RGB", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

     # Calculate processing time after the video ends
    end_processing_time = time.time()
    processing_time = end_processing_time - start_processing_time

    # Display dialog box with processing information
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    messagebox.showinfo("Processing Information", f"Processing Time: {processing_time:.2f} seconds\nAverage FPS: {avg_fps:.2f}\nMaximum Confidence Score Truck: {max_confidence:.2f}\nMaximum Confidence Score OD: {max_confidence_od:.2f}")

    # Save processing information to a text file
    with open("result.txt", "a") as file:
        file.write(f"#### YOLOv8 ####\n")
        file.write(f"Processing Time: {processing_time:.2f} seconds\n")
        file.write(f"Average FPS: {avg_fps:.2f}\n")
        file.write(f"Maximum Confidence Score Truck: {max_confidence:.2f}\n")
        file.write(f"Maximum Confidence Score OD: {max_confidence_od:.2f}\n\n")