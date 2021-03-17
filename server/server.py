
import io
import json
import os

import config
import model
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def init():
    global net
    net = model.HotDogNotHotDogClassifier().to(config.DEVICE)
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'hdnhdfier2.pth')
    net.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    net.eval()

def run(raw_data):
    # image = Image.open(io.BytesIO(json.loads(raw_data)['data']))
    json_input = json.loads(raw_data)
    data = np.array(json_input["data"], dtype=np.uint8)
    data = data.reshape(json_input["shape"])
    img = Image.fromarray(data)
    tensor =  config.IMAGE_TRANFORM_INFERENCE(img).unsqueeze(0)
    tensor = tensor.to(config.DEVICE)
    outputs = net(tensor)
    score = outputs.item()
    if score > 0.5:
        return {"class": 'Hot Dog', "score": score}
    else:
        return {"class": 'Not Hot Dog', "score": 1 - score}
