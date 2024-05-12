import numpy as np
import cv2
from common import global_function as utils


def start(img):
# Read classes from class.txt file
    net = cv2.dnn.readNetFromONNX("YOLOv6/config/od.onnx")
    with open("YOLOv6/config/od_class.txt", "r") as f:
        classes = f.read().strip().split("\n")

    # initialize image without bbox
    image_ori = utils.frameToImg(img)

    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255, size=[640, 640], mean=[0, 0, 0], swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()[0]

    classes_ids = []
    confidences = []
    boxes = []
    rows = detections.shape[0]
    conf = 0.0

    img_width, img_height = img.shape[1], img.shape[0]
    x_scale = img_width / 640
    y_scale = img_height / 640

    for detection in range(rows):
            row = detections[detection]
            if row[4] > 0.2:
                scores = row[5:]
                classId = np.argmax(scores)
                confidence = scores[classId] * row[4]
                if confidence > 0.2:
                    cx, cy, w, h = row[:4]
                    x1 = int((cx - w / 2) * x_scale)
                    y1 = int((cy - h / 2) * y_scale)
                    width = int(w * x_scale)
                    height = int(h * y_scale)

                    confidences.append(float(confidence))
                    boxes.append([x1, y1, width, height])
                    classes_ids.append(classId)

    # for i in range(rows):
    #     row = detections[i]
    #     confidence = row[4]
    #     if confidence > 0.5:
    #         classes_score = row[5:]
    #         ind = np.argmax(classes_score)
    #         if classes_score[ind] > 0.2:
    #             classes_ids.append(ind)
    #             confidences.append(confidence)
    #             cx, cy, w, h = row[:4]
    #             x1 = int((cx - w / 2) * x_scale)
    #             y1 = int((cy - h / 2) * y_scale)
    #             width = int(w * x_scale)
    #             height = int(h * y_scale)
    #             box = np.array([x1, y1, width, height])
    #             boxes.append(box)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.2)

    for i in indices:
        x1, y1, w, h = boxes[i]
        label = classes[classes_ids[i]]
        conf = confidences[i]
        text = label + "{:.2f}".format(conf)
        cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)
        cv2.putText(img, text, (x1, y1 - 2), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

        if label == 'od':
            image_bbox = utils.frameToImg(img)
            payload = {
                'image_ori': image_ori,
                'image_bb': image_bbox
            }
            response = utils.upload(payload)
            cv2.putText(img, f'Label: {response.text}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Deteksi OD", img)    

    return conf