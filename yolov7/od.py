import torch
import cv2
import numpy as np
from numpy import random
from yolov7.utils.general import non_max_suppression, scale_coords, xyxy2xywh
from yolov7.utils.plots import plot_one_box
from yolov7.utils.datasets import letterbox
from common import global_function

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model():
    weights = torch.load('yolov7/config/od.pt')
    model = weights['model']
    model = model.float().to(device)
    if torch.cuda.is_available():
        model.half()
    return model

def start(image):
    model = load_model()

    image_ori = global_function.frameToImg(image)
    frame = image

    # Get class names
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

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
                plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=2)
                if names[int(cls)] == 'od':
                    image_bb = global_function.frameToImg(frame)
                    # Prepare payload
                    payload = {'image_ori': image_ori, 'image_bb': image_bb}
                    response = global_function.upload(payload)
                    cv2.putText(frame, f'Label: {response.text}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Deteksi OD', frame)
                