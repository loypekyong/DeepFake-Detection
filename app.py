import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
#import tensorflow dataloader
from tensorflow.keras.preprocessing import image




st.title('DeepFake Detection')

#upload image and show
test_image = st.file_uploader("Upload test image", type="jpg")

if test_image is not None:

    st.image(test_image, caption='Uploaded Image.', use_column_width=True)

    # Load the image
    img = plt.imread(test_image)
    st.write(img.shape)


st.write("EfficientNetB0")
efficientnetb0 = load_model('C:/Users/loype/OneDrive - Singapore University of Technology and Design/Term 6/Artificial Intelligence/DeepFake-Detection/EfficientNetb0.keras')

image = image.load_img(test_image, target_size=(224, 224))

efficientnetb0_pred = efficientnetb0.predict(test_image)

st.write(efficientnetb0_pred)

st.write("EfficientNetv2B0")
