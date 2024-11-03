import streamlit as st
#------- needed libraries -----------------#
import matplotlib.pyplot as plt 
import numpy as np 
import PIL
from PIL import Image
import importlib
import zipfile
import os 
import gdown

from tensorflow.keras.applications import (ResNet50,EfficientNetB1, VGG16, InceptionV3, DenseNet121, MobileNetV2)
import tensorflow as tf 

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models


# import efficientnet.tfkeras as efn  # Use this import


import warnings
warnings.filterwarnings('ignore')


#------------------------------------#
with st.sidebar:
    st.title("ModelMancer")
    st.image("robo.png",'Hi there!')
    st.write("Embark on a journey through the magic of pretrained models.")
    step = st.selectbox('Select step',options=['Model Predictions','Fine Tunning'])


# ---------------------------------#
# Dictionary of available models (customize with the full list if needed)
available_models = {
    "ResNet50": ResNet50,
    "EfficientNetB0": EfficientNetB1,
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
    "EfficientNetB0": "tensorflow.keras.applications.EfficientNetB1",
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


    # make user upload data 
    uploaded_img = st.file_uploader("Upload an image for prediction",type=['jpg','png','jpeg'])
    
    # # check if there is image in data , if so use it for prediction 
    # if os.path.exists(img_path):
    #     st.write("image in files are used")
    #     img = image_preprocess(img_path,model_name)

    try : 
        # upload image by user 
        if  uploaded_img:
            img = Image.open(uploaded_img)
            img = image_preprocess(uploaded_img,model_name)

        # Predict and decode results
        predictions = model.predict(img)
        decoded_predictions = decode_predictions(predictions,top=3)[0]
        
        # show top predictions
        st.write('Top Predictions:')
        for i,(imagenet_id,label,score) in enumerate(decoded_predictions):
            st.write(f"{i+1} : {label}({score:0.2f})")
        st.image(uploaded_img, caption="Uploaded Image")
    except Exception as e :
        st.warning("Please upload an image file to proceed.")
#------------------------------------------------------------------------ # 
if step == 'Fine Tunning':
    #-------------- Uploading Data-------------- #
    ## train data images 
    st.title('Uploading Data Images:')


    # check if images will be uploaded from local folder
    upload_option = st.radio("Select Uploading method", options=['Locally','Cloud\Drive'])


    # [FF] 
    # function for uploading Data
    def uploading_data(type,clss_mode,shuff):
        st.subheader(f'Upload {type} Data',divider=True)
        data_path =None

        try: 
            if upload_option == "Locally": 
                uploaded_file=st.file_uploader(f"Upload your zipped folder of {type} images", type='zip') 
            elif upload_option == "Cloud\Drive": 
                file_id  = st.text_input("Enter your Google Drive file ID for the zipped images:")
                if file_id :
                    url = f"https://drive.google.com/uc?id={file_id}"    # Use gdown to download the file
                    uploaded_file = "images.zip"
                    st.write("Downloading file ...")
                    gdown.download(url=url,output=uploaded_file,quiet=False) # download files locally 

            # if a file uploaded , extract it 
            if uploaded_file is not None :
                # Extract the downloaded zip file
                with zipfile.ZipFile(uploaded_file,'r') as zip_ref:
                    zip_ref.extractall(f"{type}_data") # extract to a folder named Train data
                data_path =f"{type}_data" # Root directory after extraction
            
        except Exception as e:
                st.error('error uploading file')  


        # Load images if path is valid
        if data_path:
            datagen= ImageDataGenerator(rescale=1.0/255.0) # Normalize pixel values
            dataset = datagen.flow_from_directory(   # load and preprocess data 
                directory =data_path,
                target_size=(224,224), # Resize images to match model input
                batch_size = 32, # Batch size for training
                class_mode = clss_mode, # 'binary' or 'categorical' based on labels
                shuffle = shuff
            )

            st.info(f"{type.capitalize()} Data Loaded Successfully")
            return dataset
        else : 
            st.warning("Please Upload Data File to proceed.")
            return None

    # upload traing data
    training_dataset= uploading_data(type='train',clss_mode='categorical',shuff=True)
    validation_dataset = uploading_data(type='validation',clss_mode='categorical',shuff=True)
    st.divider()

    # ----------------- [FF] -------------------- #
    # function for selecting pretrained model             
    def select_pretrained_model(available_models):
        model_name = st.selectbox('select model',options=list(available_models.keys()))
        st.info('Loading model')

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
    

    # ---------------- decoding function --------------#
    def my_decoding(training_dataset,predictions):
        # Mapping output index to your class names
        class_map =  training_dataset.class_indices
        # Create a reverse mapping from index to class label
        reverse_class_map = {v: k for k, v in class_map.items()}

        # Get predicted class index
        predicted_class_index = np.argmax(predictions[0])

        # Get the class label from the reverse mapping
        predicted_class_label  = reverse_class_map[predicted_class_index]
        return st.info(f"Predicted Class:--> {predicted_class_label}")
    
    # ----------------- Modelling -------------------- #
    # model selection 
    try : 
        st.subheader("Select Pretreianed Model",divider=True)
        # Load the base model without the top layer for transfer learnign 
        base_model,preprocess_input,decode_predictions,model_Name = select_pretrained_model(available_models)
        base_model.trainable = False # Freeze the base model layers [do not train the base model]


        # Add custom classification layers
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128,activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(training_dataset.class_indices),activation='softmax') # softmax [catigorical data]
        ])
        st.subheader("Model Compilling",divider=True)
        # compile the model
        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
            loss = 'categorical_crossentropy',
            metrics = ['accuracy']
        )
        st.success('Model compiled Successfully')
    except Exception as e:
        st.warning("Model couldn't be compiled!")

    #-------------- Train Model-------------- #
    # Fit the Model
    try: 
        st.subheader("Model Training",divider=True)
        history = model.fit(
            training_dataset,
            validation_data = validation_dataset,
            epochs = 10
        )
        st.success('Model Trained Successfully')
    except Exception as e:
        st.warning("Model couldn't be Trained!")

    #-------------- Evaluate Model-------------- #
    try:
        st.subheader("Model Evaluation")
        loss , accuracy = model.evaluate(validation_dataset)
        st.write(f"Validation Accuracy: {accuracy:0.2f}")
        st.write(f'Loss : {loss:0.2f}')
    except Exception as e :
        st.warning("Model couldn't be Evaluated!")
    #-------------- Test Model with uploaded image -------------- #
    try:
        st.subheader( "Model Testing",divider=True)

        # make user upload data 
        test_img = st.file_uploader("Upload an image for prediction",type=['jpg','png','jpeg'])
        # check if there is image in data , if so use it for prediction 
        if test_img:
            test_image = image_preprocess(test_img, model_Name)
            test_image = preprocess_input(test_image)


        # Predict and decode results
        predictions = model.predict(test_image)
        st.info(f"Confidence Score: {np.max(predictions[0]):.2f}")
        prediction = np.argmax(predictions[0])

        my_decoding(training_dataset,predictions)

        im = Image.open(test_img)
        st.image(image=im,caption='Test Image')
    except Exception as e :
        pass
    st.balloons()
    