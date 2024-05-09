import numpy as np
import cv2
import time

def start(source_path, od_function):
    cap = cv2.VideoCapture(source_path)
    # load model and classes
    net = cv2.dnn.readNetFromONNX("YOLOv6/config/truck.onnx")
    with open("YOLOv6/config/truck_class.txt", "r") as f:
        classes = f.read().strip().split("\n")

    # Set the line position
    line_position = 420  # Adjust this value based on your video dimensions

    # Initialize variables for FPS calculation
    fps_values = []
    avg_fps = 0

    # processing od status
    processed_od = False

    while True:
        # Measure the start time to calculate FPS
        start_time = time.time()

        _, img = cap.read()

        # Get the current size of the frame
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Print the current size of the frame
        # print("Frame Width:", frame_width)
        # print("Frame Height:", frame_height)

        # # Resize the frame to 640x640
        # img = cv2.resize(img, (900, 600))

        blob = cv2.dnn.blobFromImage(img, scalefactor=1/255, size=[640, 640], mean=[0, 0, 0], swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward()[0]

        classes_ids = []
        confidences = []
        boxes = []
        rows = detections.shape[0]

        img_width, img_height = img.shape[1], img.shape[0]
        x_scale = img_width / 640
        y_scale = img_height / 640

        for i in range(rows):
            row = detections[i]
            confidence = row[4]
            if confidence > 0.2:
                classes_score = row[5:]
                ind = np.argmax(classes_score)
                if classes_score[ind] > 0.2:
                    classes_ids.append(ind)
                    confidences.append(confidence)
                    cx, cy, w, h = row[:4]
                    x1 = int((cx - w / 2) * x_scale)
                    y1 = int((cy - h / 2) * y_scale)
                    width = int(w * x_scale)
                    height = int(h * y_scale)
                    box = np.array([x1, y1, width, height])
                    boxes.append(box)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.2)

        for i in indices:
            x1, y1, w, h = boxes[i]
            label = classes[classes_ids[i]]
            conf = confidences[i]
            text = label + "{:.2f}".format(conf)

            # Check if the low part of the bounding box has crossed the line
            if (line_position - 6) < (y1 + h) < line_position and not processed_od:
                # set processed to true
                processed_od = True
                # Crop the object
                object_image = img[y1:y1 + h, x1:x1 + w]
                # Resize the cropped object to 640x640
                object_image_resized = cv2.resize(object_image, (640, 640))

                # Call od_detection function with the resized cropped object as parameter
                od_function(object_image_resized)

            if (y1 + h ) > (line_position + 6):
                # else, not hit the line will set not processed/false
                processed_od = False        

            cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)
            cv2.putText(img, text, (x1, y1 - 2), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

        # Draw the line
        cv2.line(img, (0, line_position), (img_width, line_position), (0, 255, 0), 2)

        # Calculate FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        fps_values.append(fps)
        avg_fps = np.mean(fps_values)

        # Draw FPS on the frame
        cv2.putText(img, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw average FPS below the FPS display
        cv2.putText(img, f'Avg FPS: {avg_fps:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw processing status
        cv2.putText(img, f'Processed: {processed_od}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Deteksi Objek", img)
        if cv2.waitKey(1) & 0xff == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
