import io
# import json

import config
import model
import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
from flask import render_template, Response
# import time
# import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--weights', type=str, required=True)
# args = parser.parse_args()

app = Flask(__name__)
net = model.HotDogNotHotDogClassifier().to(config.DEVICE)
net.load_state_dict(torch.load('model/hdnhdfier2.pth', map_location=config.DEVICE))
net.eval()

def bytes2PILImage(image_bytes):
    return Image.open(io.BytesIO(image_bytes))

def transform_image(pil_image):
    my_transforms = config.IMAGE_TRANFORM_INFERENCE
    return my_transforms(pil_image).unsqueeze(0)

def get_prediction(pil_image):
    tensor = transform_image(pil_image)
    outputs = net.forward(tensor)
    score = outputs.item()
    if score > 0.5:
        return 'Hot Dog', score
    return 'Not Hot Dog', 1-score

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # file = request.data
        # img_bytes = file.read()
        global img_bytes
        global class_name
        img_bytes = request.data
        img = bytes2PILImage(img_bytes)
        class_name, score = get_prediction(img)
        print(class_name)
        print(score)
        return jsonify({'class_name': class_name, 'score': score})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)
