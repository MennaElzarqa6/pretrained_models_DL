import streamlit as st
#------- needed libraries -----------------#
import matplotlib.pyplot as plt 
import numpy as np 
import PIL
import importlib
import zipfile
import os 
import gdown

from tensorflow.keras.applications import (ResNet50, EfficientNetB0, VGG16, InceptionV3, DenseNet121, MobileNetV2)
import tensorflow as tf 

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models

import warnings 
warnings.filterwarnings('ignore')
#------------------------------------#
with st.sidebar:
    st.title("ModelMancer")
    st.image("robo.jpg",'Hi there!')
    st.write("Embark on a journey through the magic of pretrained models.")
    step = st.selectbox('Select step',options=['Model Predictions','Fine Tunning'])
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
# ----------------------------------------------------------------------- #
# ----------- functio to preprocess the input images to the model ------- #
def image_preprocess(img_path,model_name):
        target_size = (299, 299) if model_name=='InceptionV3' else (224,224)
        img = image.load_img(img_path,target_size=target_size) # Resize image to 224x224
        img_array = image.img_to_array(img) # # Convert image to array
        img_array = np.expand_dims(img_array,axis=0) # Add patch dimension
        return preprocess_input(img_array) # preprocess to match selected model
#------------------------------------------------------------------------ # 
if step == 'Model Predictions':
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
    img_path = "apple.jpg" # use image data in files 
    # make user upload data 
    uploaded_img = st.file_uploader("Upload an image for prediction",type=['jpg','png','jpeg'])
    

    # check if there is image in data , if so use it for prediction 
    if os.path.exists(img_path):
        st.write("image in files are used")
        img = image_preprocess(img_path,model_name)

    # upload image by user 
    if  uploaded_img is not None :
        img = image_preprocess(uploaded_img,model_name)

    # Predict and decode results
    predictions = model.predict(img)
    
    # decode_predictions 
    decoded_predictions = decode_predictions(predictions,top=3)[0]
    # show top predictions
    st.write('Top Predictions:')

    for i,(imagenet_id,label,score) in enumerate(decoded_predictions):
        st.write(f"{i+1} : {label}({score:0.2f})")
#------------------------------------------------------------------------ # 
if step == 'Fine Tunning':
    # uploading data images 
    ## train data images 
    st.title('Uploading Data Images:')
    # check if images will be uploaded from local folder
    upload_option = st.checkbox("Upload from local folder")
    def uploading_data(type,clss_mode,shuff):
        st.subheader(f'Upload {type} Data',divider=True)
        try: 
            if upload_option: # local upload
                # upload zipped images file 
                uploaded_file=st.file_uploader(f"Upload your zipped folder of {type} images", type='zip')
                
            else :      # google drive upload
                # Prompt the user to enter the Google Drive file ID
                file_id  = st.text_input("Enter your Google Drive file ID for the zipped images:")
                if file_id :
                    # Use gdown to download the file
                    url = f"https://drive.google.com/uc?id={file_id}"
                    uploaded_file = "images.zip"

                    st.write("Downloading file ...")
                    gdown.download(url=url,output=uploaded_file,quiet=False)

            # if a file uploaded , extract it 
            if uploaded_file is not None :
                # Extract the downloaded zip file
                with zipfile.ZipFile(uploaded_file,'r') as zip_ref:
                    zip_ref.extractall(f"{type}_data") # extract to a folder named Train data
                data_path =f"{type}_data" # Root directory after extraction
            
        except Exception as e:
                st.error('error uploading file')  

        # Check if the path is entered
        if data_path:
            # Set up data generator for loading images from the folder
            datagen= ImageDataGenerator(rescale=1.0/255.0) # Normalize pixel values
            # Load images from the provided directory
            train_dataset = datagen.flow_from_directory(
                directory =data_path,
                target_size=(224,224), # Resize images to match model input
                batch_size = 32, # Batch size for training
                class_mode = clss_mode, # 'binary' or 'categorical' based on labels
                shuffle = shuff
            )
            st.info(f"{type} Data Loaded Successfully")
        else : 
            st.error(f"Error Loading Data: {e}")

    # upload traing data
    uploading_data(type='train',clss_mode='categorical',shuff=True)
    uploading_data(type='validation',clss_mode='categorical',shuff=True)

    # Load data (update `your_data_directory` to your dataset path)
    train_dataset =image_dataset_from_directory(
        '/workspaces/pretrained_models_DL/train_data',
        image_size = (224,224),
        batch_size = 32
    )

    validation_set = image_dataset_from_directory(
        '/workspaces/pretrained_models_DL/validation_data',
        image_size = (224,224),
        batch_size = 32
    )

    # Get Number of classes 
    num_Classes = len(train_dataset.class_names)
    st.divider()
    st.write("classes number = ",num_Classes)

    # ------------ modelling -------------------- #
    # function for selecting pretrained model 
    def select_pretrained_model(available_models):
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
        model = model_class(weights="imagenet",include_top=False) # unclude_top = False == remove last layers
        st.write(f"Loaded {model_name} model with ImageNet weights.")
        return model, preprocess_input,decode_predictions,model_name
    
    st.subheader("Select Pretreianed Model",divider=True)
    # Load the base model without the top layer
    base_model,preprocess_input,decode_predictions,model_Name = select_pretrained_model(available_models)
    base_model.trainable = False # Freeze the base model layers

    # Add custom classification layers
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128,activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_Classes,activation='sigmoid') # softmax
    ])
    st.subheader("Model Compilling",divider=True)
    # compile the model
    # st.markdown('Compilling model...')
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
        loss = 'binary_crossentropy',
        metrics = ['accuracy']
    )
    st.info('Model compiled Successfully')

    # ------- preprocess data images ----- #
    # Define a function to resize and preprocess images for MobileNetV2
    def preprocess(image,label):
        image=tf.image.resize(image, (224,224)) # Resize images to 224x224
        image = preprocess_input(image) # scale images to the range [-1,1]
        return image,label

    # preprocess image data 
    train_dataset = train_dataset.map(preprocess)
    validation_set =validation_set.map(preprocess)


    # Train the Model
    st.subheader("Model Training",divider=True)
    epochs = 10 
    history = model.fit(
        train_dataset,
        validation_data = validation_set,
        epochs = epochs
    )
    st.info('Model Trained Successfully')

    # evaluate the model
    st.subheader("Model Evaluation")
    loss , accuracy = model.evaluate(validation_set)
    st.write(f"Validation Accuracy: {accuracy:0.2f}")

    # Test the model
    st.subheader( "Model Testing",divider=True)
  

    # Function to preprocess and predict the label for a single image
    def load_and_preprocess_image(img_path):
        img = image.load_img(img_path, target_size=(224, 224))  # Resize to 224x224
        img_array = image.img_to_array(img)  # Convert to array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = preprocess_input(img_array)  # Preprocess for model selected 
        return img_array
    
     # Path to your image
    test_image_path  = 'plant.png' # use image data in files 
    # make user upload data 
    uploaded_image = st.file_uploader("Upload an image for prediction",type=['jpg','png','jpeg'])
    
    # check if there is image in data , if so use it for prediction 
    if os.path.exists(test_image_path):
        st.write("image in files are used")
        test_image = load_and_preprocess_image(test_image_path)

    # upload image by user 
    if  uploaded_image is not None :
        test_image = load_and_preprocess_image(uploaded_image)

    # # Predict and decode results
    # predictions = model.predict(test_image)
    
    # # decode_predictions 
    # predicted_class_index = np.argmax(predictions[0]) # Get the index of the highest probability class
    # # predicted_class_label = class_names[predicted_class_index] # Map index to class label
    #  # show top predictions
    # # st.write(f"Predicted Class: {predicted_class_label}")
    # st.write(f"Confidence Score: {np.max(predictions[0]):.2f}")
 
    predictions = model.predict(test_image)
    confidence_score = predictions[0][0]
    st.write(f"Confidence Score: {np.max(predictions[0]):.2f}")
    
    # Optionally, display the test image
    img = image.load_img(test_image_path)
    plt.imshow(img)
    plt.title("Prediction:{predicted_class_label}")
    plt.axis('off')
    plt.show()
