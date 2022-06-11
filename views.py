import base64
from io import BytesIO
import cv2
import numpy as np
from django.shortcuts import render
from .forms import ImageForm
import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_PATH = 'models/20211031-025213.h5'

# Load your trained model
model = load_model(MODEL_PATH, compile=False)

def predict_mask(img_path):
    H = 256
    W = 256
    num_classes = 4

    input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    input_img = cv2.resize(input_img, (W, H))
    input_img = input_img / 255.0
    input_img = input_img.astype(np.float32)

    ## Prediction
    pred_mask = model.predict(np.expand_dims(input_img, axis=0))[0]
    pred_mask = np.argmax(pred_mask, axis=-1)
    pred_mask = np.expand_dims(pred_mask, axis=-1)
    pred_mask = pred_mask * (255/num_classes)
    pred_mask = pred_mask.astype(np.int32)
    pred_mask = np.concatenate([pred_mask, pred_mask, pred_mask], axis=2)

    input_img = input_img * 255.0
    input_img = input_img.astype(np.int32)

    return input_img, pred_mask

def to_data_uri(pil_img):
    data = BytesIO()
    pil_img.save(data, "PNG")
    data64 = base64.b64encode(data.getvalue())
    return u'data:img/png;base64,'+data64.decode('utf-8')


def index(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            img_obj = form.instance
            form.save()

            # Making Prediction on image
            input_img, pred_mask = predict_mask(
                "media/"+str(img_obj.image)
                )

            input_image = tf.keras.preprocessing.image.array_to_img(
                input_img)
            input_image_uri = to_data_uri(input_image)

            mask_image = tf.keras.preprocessing.image.array_to_img(pred_mask)
            mask_uri = to_data_uri(mask_image)

            context = {"form": form, 'img_obj': img_obj, 'input_image_uri': input_image_uri,
                       'mask_uri': mask_uri}

            return render(request, 'Home/index.html', context = context)

    else:
        form = ImageForm()
    return render(request, 'Home/index.html', {"form": form})
