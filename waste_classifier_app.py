# import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array, load_img
# import numpy as np
# import os

# # Set the title of the web app
# st.title('Waste Image Classification')

# # Define the class labels with detailed mappings
# class_labels = {
#     0: 'aerosol_cans', 1: 'aluminum_food_cans', 2: 'aluminum_soda_cans', 3: 'cardboard_boxes',
#     4: 'cardboard_packaging', 5: 'clothing', 6: 'coffee_grounds', 7: 'disposable_plastic_cutlery',
#     8: 'eggshells', 9: 'food_waste', 10: 'glass_beverage_bottles', 11: 'glass_cosmetic_containers',
#     12: 'glass_food_jars', 13: 'magazines', 14: 'newspaper', 15: 'office_paper',
#     16: 'paper_cups', 17: 'plastic_cup_lids', 18: 'plastic_detergent_bottles', 19: 'plastic_food_containers',
#     20: 'plastic_shopping_bags', 21: 'plastic_soda_bottles', 22: 'plastic_straws', 23: 'plastic_trash_bags',
#     24: 'plastic_water_bottles', 25: 'shoes', 26: 'steel_food_cans', 27: 'styrofoam_cups',
#     28: 'styrofoam_food_containers', 29: 'tea_bags'
# }

# # Load the pre-trained model
# @st.cache(allow_output_mutation=True)
# def get_model():
#     model_path = 'models/waste_classification_model_combined_ResNet50.keras'
#     model = load_model(model_path)
#     return model

# model = get_model()

# # Define image preprocessing function
# def preprocess_image(image):
#     img = load_img(image, target_size=(256, 256))
#     img_array = img_to_array(img)
#     img_array_expanded_dims = np.expand_dims(img_array, axis=0)
#     return tf.keras.applications.resnet50.preprocess_input(img_array_expanded_dims)

# # Create a file uploader to upload images
# uploaded_files = st.file_uploader("Choose images to classify", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

# if uploaded_files:
#     # Display uploaded images
#     for uploaded_file in uploaded_files:
#         st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
#         # Process each image and predict
#         preprocessed_image = preprocess_image(uploaded_file)
#         predictions = model.predict(preprocessed_image)
#         predicted_class = np.argmax(predictions, axis=1)
#         # Display the prediction with class label
#         st.write(f'Prediction: {class_labels[predicted_class[0]]}')

# st.sidebar.title("About the App")
# st.sidebar.info("This is a simple image classification web app to classify waste images into 30 different categories using a ResNet50 model.")

# st.sidebar.title("How to use")
# st.sidebar.info("Upload the images of the waste items and get the classification results instantly!")

# # Optional: add a footer
# st.markdown("""
# <style>
# .footer {
#     position: fixed;
#     left: 0;
#     bottom: 0;
#     width: 100%;
#     text-align: center;
# }
# </style>
# <div class="footer">
# <p>Developed with Streamlit</p>
# </div>
# """, unsafe_allow_html=True)

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Set the title of the web app
st.title('Waste Image Classification')

# Define the class labels with detailed mappings
class_labels = {
    0: 'aerosol_cans', 1: 'aluminum_food_cans', 2: 'aluminum_soda_cans', 3: 'cardboard_boxes',
    4: 'cardboard_packaging', 5: 'clothing', 6: 'coffee_grounds', 7: 'disposable_plastic_cutlery',
    8: 'eggshells', 9: 'food_waste', 10: 'glass_beverage_bottles', 11: 'glass_cosmetic_containers',
    12: 'glass_food_jars', 13: 'magazines', 14: 'newspaper', 15: 'office_paper',
    16: 'paper_cups', 17: 'plastic_cup_lids', 18: 'plastic_detergent_bottles', 19: 'plastic_food_containers',
    20: 'plastic_shopping_bags', 21: 'plastic_soda_bottles', 22: 'plastic_straws', 23: 'plastic_trash_bags',
    24: 'plastic_water_bottles', 25: 'shoes', 26: 'steel_food_cans', 27: 'styrofoam_cups',
    28: 'styrofoam_food_containers', 29: 'tea_bags'
}

# Load the pre-trained model
@st.cache_resource()
def get_model():
    model_path = 'models/waste_classification_model_comb_InceptionV3.keras'
    model = load_model(model_path)
    return model

model = get_model()

# Define image preprocessing function
def preprocess_image(image):
    img = load_img(image, target_size=(256,256 ))
    img_array = img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.resnet50.preprocess_input(img_array_expanded_dims)

# Create a file uploader to upload images
uploaded_files = st.file_uploader("Choose images to classify", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

if uploaded_files:
    # Display uploaded images
    for uploaded_file in uploaded_files:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        # Process each image and predict
        preprocessed_image = preprocess_image(uploaded_file)
        predictions = model.predict(preprocessed_image)
        predicted_class = np.argmax(predictions, axis=1)
        predicted_proba = np.max(predictions, axis=1)
        
        # Debug: print raw predictions
        # st.write(f'Raw predictions: {predictions}')
        # st.write(f'Predicted class index: {predicted_class[0]}')
        st.write(f'Predicted probability: {predicted_proba[0]}')

        # Display the prediction with class label
        st.write(f'Prediction: {class_labels[predicted_class[0]]}')

st.sidebar.title("About the App")
st.sidebar.info("This is a simple image classification web app to classify waste images into 30 different categories using a InceptionV3 model.")
st.sidebar.title("How to use")
st.sidebar.info("Upload the images of the waste items and get the classification results instantly!")

# Optional: add a footer
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    text-align: center;
}
</style>
<div class="footer">
<p>Developed with Streamlit</p>
</div>
""", unsafe_allow_html=True)
