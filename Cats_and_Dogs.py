#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import time

# Define the Streamlit app
def app():
    if "X" not in st.session_state: 
        st.session_state.X = []
    
    if "y" not in st.session_state: 
        st.session_state.y = []

    if "model" not in st.session_state:
        st.session_state.model = []

    if "X_train" not in st.session_state:
        st.session_state.X_train = []

    if "X_test" not in st.session_state:
            st.session_state.X_test = []

    if "y_train" not in st.session_state:
            st.session_state.y_train = []

    if "y_test" not in st.session_state:
            st.session_state.y_test = []

    if "X_test_scaled" not in st.session_state:
            st.session_state.X_test_scaled = []

    if "n_clusters" not in st.session_state:
        st.session_state.n_clusters = 4

    text = """Convolutional Neural Network Image Classifier"""
    st.subheader(text)

    text = """Prof. Louie F. Cervantes, M. Eng. (Information Engineering) \n
    CCS 229 - Intelligent Systems
    Department of Computer Science
    College of Information and Communications Technology
    West Visayas State University"""
    st.text(text)

    st.image('cat_or_dog.jpg', caption='Cat or Dog Image Classification')

    text = """
This Streamlit app demonstrates a binary image classifier for cats and dogs 
using a Convolutional Neural Network (CNN). The CNN is trained on a balanced 
dataset of 4,000 images, containing 2,000 cat and 2,000 dog images. The app leverages 
Streamlit's capabilities to create a user-friendly interface for image 
uploading and classification. Upon uploading an image, the app pre-processes it 
(resizing, normalization) and feeds it through the trained CNN model. 
The model's output is then interpreted to predict whether the image contains 
a cat or a dog.
    """
    st.write(text)

    with st.expander("How to use this App"):
         text = """Step 1. Go to Training page. Set the parameters of the CNN. Click the button to begin training.
         \nStep 2.  Go to Performance Testing page and click the button to load the image
         and get the model's output on the classification task.
         \nYou can return to the training page to try other combinations of parameters."""
         st.write(text)
    
#run the app
if __name__ == "__main__":
    app()