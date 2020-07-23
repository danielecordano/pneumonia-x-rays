# -*- coding: utf-8 -*-

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps


def image_from_data(image_data):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = (image.astype(np.float32) / 255.0)
    img_reshape = image[np.newaxis, ...]
    return img_reshape


st.write("""# Pneumonia detector""")
st.write("This is a simple image classification web app to detect pneumonia from chest X-rays.")

file = st.file_uploader("Please upload an image file",
                        type=["jpg", "jpeg", "png"])

if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    img = image_from_data(image)
    model = tf.keras.models.load_model('pneumonia_mobilenet_v2')
    prediction = model.predict(img)
    threshold = 0.69613934
    if prediction < threshold:
        st.write("Your chest X-rays seem OK")
    else:
        st.write("Pneumonia was detected")
    st.text("0: Normal, 1: Pneumonia")
    st.write(prediction)
