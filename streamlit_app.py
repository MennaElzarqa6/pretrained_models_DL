import streamlit as st
#------- needed libraries -----------------#
import matplotlib.pyplot as plt 
import numpy as np 
import PIL
import importlib
import os 

from tensorflow.keras.applications import (ResNet50, EfficientNetB0, VGG16, InceptionV3, DenseNet121, MobileNetV2)
import tensorflow as tf 

from tensorflow.keras.preprocessing import image

import warnings 
warnings.filterwarnings('ignore')
#------------------------------------#
with st.sidebar:
    st.title("ModelMancer")
    st.image("robo.jpg",'Hi there!')
    st.write("Embark on a journey through the magic of pretrained models.")
    step = st.selectbox('Select step',options=['Model Selection','modelling','evaluating'])
# ---------------------------------#
# Dictionary of available models (customize with the full list if needed)
available_models = {
    "ResNet50": ResNet50,
    "EfficientNetB0": EfficientNetB0,
    "VGG16": VGG16,
    "InceptionV3": InceptionV3,
    "DenseNet121": DenseNet121,
    "MobileNetV2": MobileNetV2
}

# Define available models with their respective paths
model_options = {
    "ResNet50": "tensorflow.keras.applications.resnet50",
    "VGG16": "tensorflow.keras.applications.vgg16",
    "InceptionV3": "tensorflow.keras.applications.inception_v3",
    "EfficientNetB0": "tensorflow.keras.applications.efficientnet",
    "DenseNet121": "tensorflow.keras.applications.densenet",
    "MobileNetV2": "tensorflow.keras.applications.mobilenet_v2"
}
#------------------------------------------------------------------------ # 
if step == 'Model Selection':
    model_name = st.selectbox('select model',options=list(available_models.keys()))
    st.info('Loading model')

    # Arguments for customization

    # model importings 
    model_module = importlib.import_module(model_options[model_name])
    # Access the model, preprocess_input, and decode_predictions from the module
    model_class = getattr(model_module,model_name)
    preprocess_input = getattr(model_module, 'preprocess_input')
    decode_predictions = getattr(model_module, 'decode_predictions')

    # Initialize the model with ImageNet weights
    model = model_class(weights="imagenet")
    st.write(f"Loaded {model_name} model with ImageNet weights.")

    # Load and preprocess the image
    def image_preprocess(img_path):
        target_size = (299, 299) if model_name=='InceptionV3' else (224,224)
        img = image.load_img(img_path,target_size=target_size) # Resize image to 224x224
        img_array = image.img_to_array(img) # # Convert image to array
        img_array = np.expand_dims(img_array,axis=0) # Add patch dimension
        return preprocess_input(img_array) # preprocess to match selected model
    # Path to your image
    img_path = "aple.jpg"
    uploaded_img = st.file_uploader("Upload an image for prediction",type=['jpg','png','jpeg'])
    uploaded_img = image_preprocess(uploaded_img)
    
    if os.path.exists(img_path):
        uploaded_img = image_preprocess(img_path)
    
    if  uploaded_img is not None :
        # Predict and decode results
        predictions = model.predict(uploaded_img)
    
    # decode_predictions 
    decoded_predictions = decode_predictions(predictions,top=3)[0]
    # show top predictions
    st.write('Top Predictions:')

    for i,(imagenet_id,label,score) in enumerate(decoded_predictions):
        st.write(f"{i+1} : {label}({score:0.2f})")

