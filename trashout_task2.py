#!/usr/bin/env python
# coding: utf-8
#-------all required packages
import streamlit as st
import numpy as np
import pandas as pd
from sys import getsizeof
#from streamlit_folium import folium_static
#import folium
import requests
import matplotlib.pyplot as plt
import matplotlib.colors as colors
#from streamlit_folium import folium_static
import PIL
from PIL import Image
import requests
#------------------------------------
import tensorflow as tf
from tensorflow import one_hot
from tensorflow.keras.callbacks import LambdaCallback, Callback, ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.models import load_model
from tensorflow.keras import Model, Sequential
from tensorflow.keras import backend, regularizers
from tensorflow.keras.layers import Activation, Reshape, Dense, Embedding, Dropout, Input, BatchNormalization, concatenate, Flatten, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad, Adamax
#from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical, plot_model
from shapely.geometry import Point
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss, confusion_matrix
# from venues_deployed import getnearbyvenues
# from getnearbyvenues import *
from mrcnn.infer import get_model,predict
  
from PIL import Image, ImageDraw, ImageFont
import os

st.set_option('deprecation.showfileUploaderEncoding', False)
pd.set_option('display.max_colwidth', None)

@st.cache(persist=True, allow_output_mutation=True, suppress_st_warning=True)
def process(image, server_url: str):
    
    m = MultipartEncoder(fields={'image':('filename', image, 'image/jpeg')})
    r = requests.post(server_url,data=m,headers={'Content-Type': m.content_type},timeout=8000)
    return r
@st.cache(persist=True, allow_output_mutation=True, suppress_st_warning=True)   

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

st.write('Trashout Shopping Product Classifier!')
INCEP_LABEL_PATH = "incep_cl/trained_labels.txt"
INCEP_MODEL_PATH = "incep_cl/trained_graph_4000.pb"

# MASKRCNN_MODEL_WEIGHTS = "mrcnn/weights/mask_rcnn_trashout_0250.h5"


option_model_first = st.sidebar.radio("Choose Option",options = ['Run product classifier','Regulations information'])

#Loads label file, strips off carriage return
label_lines = [line.rstrip() for line in tf.io.gfile.GFile(INCEP_LABEL_PATH)]
#Unpersists graph from file
with tf.compat.v1.gfile.FastGFile(INCEP_MODEL_PATH, "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name="")

if option_model_first == 'Run product classifier':  
    load_image = uploaded_file = st.file_uploader("Load image", type=['jpg','png','jpeg'])
    if load_image is not None:
        image = Image.open(load_image)
        st.image(image, caption='Uploaded Image.', width = 160) 
        #st.set_option('deprecation.showfileUploaderEncoding', False)
#         option_model = st.selectbox("Choose Model" , options = ['Select Model','Inception','MaskRcnn'])
        option_model = st.selectbox("Choose Model" , options = ['Select Model','Inception'])
        if (option_model == "Inception"):
            if st.button('Get Classification'):
                image = Image.open(load_image)
                load_image = image.convert('RGB')
                load_image.save('im.jpg', format="JPEG")
    
                image_data = tf.compat.v1.gfile.FastGFile('im.jpg', "rb").read()
                with tf.compat.v1.Session() as sess:             
                    # Feed the image_data as input to the graph and get first prediction
                    st.spinner=("Inference running on the Image by model...")
                    softmax_tensor = sess.graph.get_tensor_by_name("final_result:0")

                    predictions = sess.run(softmax_tensor, {"DecodeJpeg/contents:0": image_data})

                    # Sort to show labels of first prediction in order of confidence
                    top_k = predictions[0].argsort()[-len(predictions[0]) :][::-1]
                    answer = label_lines[top_k[0]]
                    print(answer)    
                image_label = answer
                #image_label = 'Felxible Plastic'
                st.image([image], image_label, width=160)  # output dyptich

#         elif (option_model == "MaskRcnn"):
#             if st.button('Get Classification'):
#                 image = Image.open(load_image)
#                 load_image = image.convert('RGB')
#                 load_image.save('im.jpg', format="JPEG")
#                 model =get_model(MASKRCNN_MODEL_WEIGHTS,'cpu')
#                 cats=predict('im.jpg',model)
#                 answer = cats[0]
#                 print(answer)
#                 image_label = answer
#                 #image_label = 'Felxible Plastic'
#                 st.image([image], image_label, width=160)  # output dyptich

        
if option_model_first == 'Regulations information':
    final_array = []
    cg_data = pd.read_csv("communityGuidelines.csv",   engine='python')
    req_cols= ['City', 'Item Description', 'Disposal Category', 'Disposal Instructions']
    df = cg_data[req_cols]
    cities = ( "Bangalore","Toronto", "Vancouver", "Angers" )
    city= st.selectbox("Select city", cities)
    image_label_df= df.drop_duplicates(subset='Disposal Category')
    list_imagelabel = image_label_df['Disposal Category'].tolist()
    select_dc = st.selectbox("Select object" , options = list_imagelabel)
    inds = df[(df["City"]==city) &  (select_dc == df["Disposal Category"])]
    if(inds.empty):
        st.write(city, 'guideline for disposing ',select_dc, 'is not available')
    else:
        disposal_instr = inds.iloc[0]['Disposal Instructions']
        if(len(disposal_instr)==0 ):
            st.write(city, 'guideline for disposing ',select_dc, 'is not available')
        else:
            st.write(city, 'guideline for disposing ',select_dc,': ', str(disposal_instr))







