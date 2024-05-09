import cv2
import torch
import time
import numpy as np
from numpy import random
from yolov7.utils.general import non_max_suppression, scale_coords, xyxy2xywh
from yolov7.utils.plots import plot_one_box
from yolov7.utils.datasets import letterbox

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model():
    weights = torch.load('yolov7/config/truck.pt')
    model = weights['model']
    model = model.float().to(device)
    if torch.cuda.is_available():
        model.half()
    return model

def start(source_path, od_function):
    model = load_model()

    # Get class names
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Open video capture
    cap = cv2.VideoCapture(source_path)

    # Set the line position
    line_position = 420  # Adjust this value based on your video dimensions

    total_fps = 0.0
    num_frames = 0

    start_time = time.time()  # Initialize start_time

    object_detected = False  # Flag to track if object has been detected

    count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        count += 1
        if count % 3 != 0:
            continue

        frame = cv2.resize(frame, (900, 720))
        img = letterbox(frame, 640, stride=64, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    # Check if the low part of the bounding box has the same position as the line
                    if (line_position - 10) < int(xyxy[3]) and names[int(cls)] == 'truckA' and not object_detected:
                        # set object detected True
                        object_detected = True
                        # Export the cropped image
                        object_image = frame[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                        object_image_resized = cv2.resize(object_image, (640, 640))
                        od_function(object_image_resized)

                    if int(xyxy[3]) > (line_position + 6):
                        object_detected = False
                    
                    plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=2)

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
        cv2.putText(frame, f'Avg FPS: {avg_fps:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Status: {object_detected}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("YOLOv7 Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()