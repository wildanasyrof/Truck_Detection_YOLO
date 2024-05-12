import cv2
import pandas as pd
from ultralytics import YOLO
from common import global_function as utils

def start(image):

    model = YOLO('YOLOv8/config/od.pt')

    class_file = open("YOLOv8/config/od_class.txt", "r").read()
    class_list = class_file.split("\n")

    # initialize image without bbox
    image_ori = utils.frameToImg(image)

    src = image
    results = model.predict(src)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    height, width, channels = src.shape

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

        # Draw bounding box
        cv2.rectangle(src, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Draw class name
        text = f"{c}"
        cv2.putText(src, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if text == 'od':
            image_bbox = utils.frameToImg(src)
            payload = {
                'image_ori': image_ori,
                'image_bb': image_bbox
            }
            response = utils.upload(payload)
            cv2.putText(src, f'Label: {response.text}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Deteksi OD', src)

    return confidence

