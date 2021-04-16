import base64
import numpy as np
import os
import sys
import io
from PIL import Image
from keras_yolo3.yolo import YOLO, detect_video, detect_webcam
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from flask import request
from flask import render_template,redirect, url_for
from flask import jsonify
from flask import send_file
from flask import Flask

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def get_parent_dir(n=1):
    """returns the n-th parent dicrectory of the current
    working directory"""
    current_path = os.path.dirname(os.path.abspath(__file__))
    for _ in range(n):
        current_path = os.path.dirname(current_path)
    return current_path


src_path = os.path.join(get_parent_dir(1), "2_Training", "src")
utils_path = os.path.join(get_parent_dir(1), "Utils")

sys.path.append(src_path)
sys.path.append(utils_path)

from utils import load_extractor_model, load_features, parse_input, detect_object

# Set up folder names for default values
data_folder = os.path.join(get_parent_dir(1), "Data")

image_folder = os.path.join(data_folder, "Source_Images")

image_test_folder = os.path.join(image_folder, "Test_Images")

detection_results_folder = os.path.join(image_folder, "Test_Image_Detection_Results")
detection_results_file = os.path.join(detection_results_folder, "Detection_Results.csv")

model_folder = os.path.join(data_folder, "Model_Weights")

model_weights = os.path.join(model_folder, "trained_weights_final.h5")
model_classes = os.path.join(model_folder, "data_classes.txt")

anchors_path = os.path.join(src_path, "keras_yolo3", "model_data", "yolo_anchors.txt")

FLAGS = None

print(model_folder)
app = Flask(__name__)

print("*****************")
print(model_folder+"\n"+model_classes+"\n"+anchors_path)

# define YOLO detector
def get_model():
    global yolo
    yolo = YOLO(
        **{
            "model_path": model_weights,
            "anchors_path": anchors_path,
            "classes_path": model_classes,
            "score": 0.05,
            "gpu_num": 1,
            "model_image_size": (416, 416),
        }
    )

def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    #image = image.resize(target_size)
    #image = img_to_array(image)
    #image = np.expand_dims(image, axis=0)
    return image

@app.route("/",methods=["GET","POST"])
def index():
    if(request.method == "POST"):
        print(request.url)
        return redirect(request.url)
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict(debug=False):
    if(not debug):
        imageData = None
        if ('imageData' in request.files):
            imageData = request.files['imageData']
        elif ('imageData' in request.form):
            imageData = request.form['imageData']
        else:
            imageData = io.BytesIO(request.get_data())
        image = Image.open(imageData)
    else:
        image = Image.open('../../Data/Source_Images/Test_Images/4_2020-12-06_08-39-46_large.jpg')

    processed_image = preprocess_image(image)
    prediction, new_image = yolo.detect_image(processed_image)

    file_obj = io.BytesIO()
    new_image.save(file_obj,'jpeg')
    file_obj.seek(0)
    encoded_img_data = base64.b64encode(file_obj.getvalue())

    return render_template("index.html", img_data=encoded_img_data.decode('utf-8'))
    # return send_file( file_obj,
    #         mimetype='image/jpeg',
    #         as_attachment=False,
    #         attachment_filename='response.jpg')

@app.route("/",methods=["GET"])
def getter():
    return "Hello, this is a testing URL"

print("Load model...!")
get_model()