import streamlit as st
import pandas as pd
import plotly.express as px
import keras
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
import requests
import tensorflow as tf 
import requests
import imghdr
from io import BytesIO
from urllib.request import urlopen
from PIL import Image

st.set_page_config(layout="wide")
st.title("Solar Module Classification")
st.text('This Web App is used to classify Clean vs Dusty solar module.')
st.text("Choose Your Option: 1)Img_Upload 2)Img_URL")
st.text("Choose Your Model : 1)DenseNet   2)AlexNet")

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://hgtvhome.sndimg.com/content/dam/images/hgtv/fullset/2022/8/30/0/Shutterstock_1222512460_Diyana-Dimitrova_solar-panels-in-field.jpg.rend.hgtvcom.966.644.suffix/1661871715320.jpeg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 


page = st.sidebar.selectbox('UPLOAD IMAGE OR IMGAE URL',['image_upload','image_url'])
model = st.sidebar.selectbox('CHOSSE THE MODEL',['DenseNet','AlexNet']) 
if model == 'DenseNet':
    if page == 'image_upload':
        uploaded_file = st.file_uploader("Choose a solar image...", type=["jpg","png","jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded solar image.', use_column_width=True)
            st.write("")
            st.write("Classifying...")
            model = keras.models.load_model('solar_classify_finetune_final-2.h5')
            # Create the array of the right shape to feed into the keras model
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            #image sizing
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.ANTIALIAS)
            
            #turn the image into a numpy array
            image_array = np.asarray(image)
            # Normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
            # Load the image into the array
            data[0] = normalized_image_array
            class_dictionary = np.load('class_indices.npy', allow_pickle=True).item()
            # run the inference
            probabilities = model.predict(data)
            prediction_probability = probabilities[0, probabilities.argmax(axis=1)][0]
            prediction = model.predict(data)
            class_predicted = np.argmax(probabilities, axis=1)
            inID = class_predicted[0]
            inv_map = {v: k for k, v in class_dictionary.items()}
            label = inv_map[inID]
            st.write("Predicted: {}, Confidence: {:.5f}%".format(label, prediction_probability*100))
            
    else:
        url = st.text_input("Enter Image Url:")
        if url:
            img = Image.open(urlopen(url))
            classify = st.button("classify image")
            if classify:
                st.write("")
                st.write("Classifying...")
                model = keras.models.load_model('solar_classify_finetune_final-2.h5')
                # Create the array of the right shape to feed into the keras model
                data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                #image sizing
                size = (224, 224)
                image = ImageOps.fit(img, size, Image.ANTIALIAS)
                
                #turn the image into a numpy array
                image_array = np.asarray(image)
                # Normalize the image
                normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
                # Load the image into the array
                data[0] = normalized_image_array
                class_dictionary = np.load('class_indices.npy', allow_pickle=True).item()
                # run the inference
                probabilities = model.predict(data)
                prediction_probability = probabilities[0, probabilities.argmax(axis=1)][0]
                prediction = model.predict(data)
                class_predicted = np.argmax(probabilities, axis=1)
                inID = class_predicted[0]
                inv_map = {v: k for k, v in class_dictionary.items()}
                label = inv_map[inID]
                st.write("Predicted: {}, Confidence: {:.5f}%".format(label, prediction_probability*100))
                
else:
    if page == 'image_upload':
        uploaded_file = st.file_uploader("Choose a solar image...", type=["jpg","png","jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded solar image.', use_column_width=True)
            st.write("")
            st.write("Classifying...")
            model = keras.models.load_model('Alexnet_model.h5')
            # Create the array of the right shape to feed into the keras model
            data = np.ndarray(shape=(1, 227, 227, 3), dtype=np.float32)
            #image sizing
            size = (227, 227)
            image = ImageOps.fit(image, size, Image.ANTIALIAS)
            
            #turn the image into a numpy array
            image_array = np.asarray(image)
            # Normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
            # Load the image into the array
            data[0] = normalized_image_array
            class_dictionary = np.load('class_indices.npy', allow_pickle=True).item()
            # run the inference
            probabilities = model.predict(data)
            prediction_probability = probabilities[0, probabilities.argmax(axis=1)][0]
            prediction = model.predict(data)
            class_predicted = np.argmax(probabilities, axis=1)
            inID = class_predicted[0]
            inv_map = {v: k for k, v in class_dictionary.items()}
            label = inv_map[inID]
            st.write("Predicted: {}, Confidence: {:.5f}%".format(label, prediction_probability*100))
            
            
    else:
        url = st.text_input("Enter Image Url:")
        if url:
            img = Image.open(urlopen(url))
            classify = st.button("classify image")
            if classify:
                st.write("")
                st.write("Classifying...")
                model = keras.models.load_model('Alexnet_model.h5')
                # Create the array of the right shape to feed into the keras model
                data = np.ndarray(shape=(1, 227, 227, 3), dtype=np.float32)
                #image sizing
                size = (227, 227)
                image = ImageOps.fit(img, size, Image.ANTIALIAS)
                
                #turn the image into a numpy array
                image_array = np.asarray(image)
                # Normalize the image
                normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
                # Load the image into the array
                data[0] = normalized_image_array
                class_dictionary = np.load('class_indices.npy', allow_pickle=True).item()
                # run the inference
                probabilities = model.predict(data)
                prediction_probability = probabilities[0, probabilities.argmax(axis=1)][0]
                prediction = model.predict(data)
                class_predicted = np.argmax(probabilities, axis=1)
                inID = class_predicted[0]
                inv_map = {v: k for k, v in class_dictionary.items()}
                label = inv_map[inID]
                st.write("Predicted: {}, Confidence: {:.5f}%".format(label, prediction_probability*100))


