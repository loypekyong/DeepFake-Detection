import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

st.title('DeepFake Detection')

# Upload image and show
test_image = st.file_uploader("Upload test image", type=["jpg", "jpeg"])

# Let the user select the model
model_names = {
    'Model 1: EfficientNet B0 ': 'EfficientNetb0_test1.h5',
    'Model 2: EfficientNet B0 with Regularization': 'EfficientNetb0withReg_test2.h5',
    'Model 3: EfficientNet V2 B0': 'EfficientNetv2b0_test3.h5',
    'Model 4: Xception V1': 'Xception_v1_test4.h5'
}

model_option = st.selectbox('Choose the model for prediction:', options=list(model_names.keys()))

# Define a function to load the model, this will use Streamlit's newer caching mechanism
@st.cache_resource
def load_model_wrapper(model_filename):
    model = load_model(model_filename)
    return model

if test_image is not None:
    # Display the uploaded image
    uploaded_image = Image.open(test_image)
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image for the model
    img = uploaded_image.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Load the selected model (only once per session)
    model_filename = model_names[model_option]
    model_selected = load_model_wrapper(model_filename)

    # Make a prediction
    prediction = model_selected.predict(img_array)

    # Convert the prediction to a percentage
    prediction_percentages = prediction[0] * 100

    # Display the prediction probabilities with appropriate labels
    st.write("Prediction Probabilities:")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Fake Probability", f"{prediction_percentages[0]:.2f}%")
    with col2:
        st.metric("Real Probability", f"{prediction_percentages[1]:.2f}%")
