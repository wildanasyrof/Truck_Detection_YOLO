import cv2
import requests

def frameToImg(image):
    _, img_encoded = cv2.imencode('.png', image)
    img_bytes = img_encoded.tobytes() 
    return img_bytes

def upload(payload):
    url = 'https://truck.despoinalabs.com/api/upload-image'
    response = requests.post(url, files=payload)
    return response