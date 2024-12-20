#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers

import time
classes = ['cat', 'dog']

def get_class(label):
    return classes[label]

# Define the Streamlit app
def app():

    if "classifier" not in st.session_state:
        st.session_state.classifier = []

    if "training_set" not in st.session_state:
        st.session_state.training_set = []
    
    if "test_set" not in st.session_state:
        st.session_state.test_set = []

    with st.expander("Click to display more info"):
        text = """
        \n# --- Available activation functions for hidden layers ---
        ReLU (Rectified Linear Unit): Most common default
        LeakyReLU: Variant of ReLU to address "dying ReLU" issue
        tanh (Hyperbolic Tangent): Squash values between -1 and 1
        ELU (Exponential Linear Unit): Similar to ReLU, but smoother
        SELU (Scaled ELU): Variant of ELU for self-normalizing networks

        \nA convolutional neural network (CNN) is a type of artificial 
        intelligence especially good at processing images and videos.  
        Unlike other neural networks, CNNs don't need 
        images to be pre-processed by hand. Instead, they can learn to identify features 
        themselves through a process called convolution.
        \nLayers: CNNs are built up of layers, including an input layer, convolutional 
        layers, pooling layers, and fully-connected layers.
        \nConvolutional layers: These layers use filters to identify patterns and features 
        within the image. Imagine a filter like a small magnifying glass that scans the image 
        for specific details.
        \nPooling layers: These layers reduce the complexity of the image by summarizing the 
        information from the convolutional layers.
        \nFully-connected layers: These layers work similarly to regular neural networks, 
        taking the outputs from the previous layers and using them to classify the image 
        or make predictions."""
        st.write(text)


    progress_bar = st.progress(0, text="Loading the images, please wait...")

    # Data generators
    #train_datagen = ImageDataGenerator(rescale=1.0 / 255, shear_range=0.2, horizontal_flip=True)
    #test_datagen = ImageDataGenerator(rescale=1.0 / 255, shear_range=0.2, horizontal_flip=True)

    # Data augmentation (example using ImageDataGenerator)
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    # Data preparation
    training_set = train_datagen.flow_from_directory(
        "dataset/training_set",
        target_size=(64, 64),
        batch_size=32,
        class_mode="binary",
    )
    test_set = test_datagen.flow_from_directory(
        "dataset/test_set",
        target_size=(64, 64),
        batch_size=32,
        class_mode="binary",
    )

    st.session_state.training_set = training_set
    st.session_state.test_set = test_set

    # update the progress bar
    for i in range(100):
        # Update progress bar value
        progress_bar.progress(i + 1)
        # Simulate some time-consuming task (e.g., sleep)
        time.sleep(0.01)
    # Progress bar reaches 100% after the loop completes
    st.success("Image dataset loading completed!") 

    st.subheader("Sample Training Images")
    st.write("The following are 25 sample images randomly selected from both cats and dogs image dataset.")
    # Get the data for the first 25 images in training set
    train_data = next(training_set)
    train_images, train_labels = train_data[0][0:25], train_data[1][0:25]  # Get first 25 images and labels

    # Plot the training set images
    plot_images(train_images, train_labels)
    
   # Define CNN parameters    
    st.sidebar.subheader('Set the CNN Parameters')
    options = ["relu", "leaky_relu", "tanh", "elu", "selu"]
    h_activation = st.sidebar.selectbox('Activation function for the hidden layer:', options)

    options = ["sigmoid", "softmax"]
    o_activation = st.sidebar.selectbox('Activation function for the output layer:', options)

    n_neurons = st.sidebar.slider(      
        label="Number of Neurons in the Hidden Layer:",
        min_value=32,
        max_value=1024,
        value=512,  # Initial value
        step=32
    )

    epochs = st.sidebar.slider(   
        label="Set the number epochs:",
        min_value=20,
        max_value=200,
        value=20,
        step=5
    )

    classifier = keras.Sequential(
        [
            layers.Conv2D(64, (3, 3), activation=h_activation, input_shape=(64, 64, 3)),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, (3, 3), activation=h_activation),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(n_neurons, (3, 3), activation=h_activation),
            layers.Flatten(),
            layers.Dense(128, activation=h_activation),
            layers.Dense(1, activation=o_activation),
        ]
    )

    # Compile the model (adjust optimizer and loss as needed)
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    st.session_state.classifier = classifier

    with st.expander("CLick to display guide on how to select parameters"):
        text = """ReLU (Rectified Linear Unit): This is the most common activation function used 
        in convolutional neural networks (CNNs) for hidden layers. It outputs the input 
        directly if it's positive (f(x) = x for x >= 0) and sets negative inputs to zero 
        (f(x) = 0 for x < 0). ReLU is computationally efficient, avoids the vanishing 
        gradient problem, and often leads to good performance in CNNs.
        \nSigmoid: This activation function squashes the input values between 0 and 1 
        (f(x) = 1 / (1 + exp(-x))). It's typically used in the output layer of a CNN for 
        tasks like binary classification (predicting one of two classes). 
        However, sigmoid can suffer from vanishing gradients in deep networks.
        \nAdditional Activation Function Options for Hidden Layers:
        \nLeaky ReLU: A variant of ReLU that addresses the "dying ReLU" problem where some 
        neurons might never fire due to negative inputs always being zeroed out. 
        Leaky ReLU allows a small, non-zero gradient for negative inputs 
        (f(x) = max(α * x, x) for a small α > 0). This can help prevent neurons from 
        getting stuck and improve training.
        TanH (Hyperbolic Tangent): Similar to sigmoid, TanH squashes values 
        between -1 and 1 (f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))). 
        It can sometimes be more effective than sigmoid in certain tasks due to 
        its centered output range.
        \nChoosing the Right Activation Function:
        \nThe best activation function often depends on the specific problem and 
        network architecture. Here's a general guideline:
        \nHidden Layers: ReLU is a strong default choice due to its efficiency and 
        ability to avoid vanishing gradients. Leaky ReLU can be a good alternative, 
        especially in deeper networks. TanH is also an option, but ReLU is often preferred.
        \nOutput Layer:
        \nBinary Classification: Sigmoid is commonly used here for its ability to output 
        probabilities between 0 and 1.
        \nMulti-class Classification: In this case, you'd likely use a softmax activation 
        function in the output layer, which normalizes the outputs to probabilities that 
        sum to 1 (useful for predicting one of multiple exclusive classes).
        \nExperimentation:
        \nIt's always recommended to experiment with different activation functions to see 
        what works best for your specific CNN and dataset. You can try replacing "relu" 
        with "leaky_relu" or "tanh" in the hidden layers and "sigmoid" with "softmax" 
        in the output layer (if applicable) to see if it improves performance.
        \nBy understanding these activation functions and their trade-offs, you can 
        make informed choices to optimize your CNN for the task at hand."""
        st.write(text)

    if st.button('Start Training'):
 
        progress_bar = st.progress(0, text="Training the model please wait...")
        # Train the model
        batch_size = 64
        training_set = st.session_state.training_set
        test_set = st.session_state.test_set

        # alternative configuration only
        # specify instead of batch_size
        # num_train_samples = len(training_set)  # Assuming training_set is a sequence
        # steps_per_epoch = int(num_train_samples / batch_size)

        # Train the model
        history = classifier.fit(
            training_set,
            epochs=epochs,
            validation_data=test_set,
            #steps_per_epoch=steps_per_epoch,    <-- use only if you disable batch_size
            batch_size = batch_size,
            validation_steps=10,
            callbacks=[CustomCallback()]
        )

        # Evaluate the model on the test data
        accuracy = classifier.evaluate(test_set)
        st.write("Test accuracy:", accuracy)

        # Extract loss and accuracy values from history
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        train_acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        # Create the figure with two side-by-side subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # Adjust figsize for better visualization

        # Plot loss on the first subplot (ax1)
        ax1.plot(train_loss, label='Training Loss')
        ax1.plot(val_loss, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Plot accuracy on the second subplot (ax2)
        ax2.plot(train_acc, 'g--', label='Training Accuracy')
        ax2.plot(val_acc, 'r--', label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        # Set the main title (optional)
        fig.suptitle('Training and Validation Performance')

        plt.tight_layout()  # Adjust spacing between subplots
        st.pyplot(fig)   
        
        # update the progress bar
        for i in range(100):
            # Update progress bar value
            progress_bar.progress(i + 1)
            # Simulate some time-consuming task (e.g., sleep)
            time.sleep(0.01)
        # Progress bar reaches 100% after the loop completes
        st.success("Model training completed!") 
        st.write("The model is now trained to tell dogs and cats apart. Use the sidebar to open the Performance Testing page.")

# Define a function to plot images
def plot_images(images, labels):
    fig, axs = plt.subplots(5, 5, figsize=(10, 6))  # Create a figure with subplots

    # Flatten the axes for easier iteration
    axs = axs.flatten()

    for i, (image, label) in enumerate(zip(images, labels)):
        axs[i].imshow(image)  # Use ax for imshow on each subplot
        #axs[i].set_title(f"Class: {label}")  # Use ax.set_title for title
        axs[i].set_title(get_class(int(label))) 
        axs[i].axis("off")  # Use ax.axis for turning off axis

    plt.tight_layout()  # Adjust spacing between subplots
    st.pyplot(fig)

# Define a custom callback function to update the Streamlit interface
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Get the current loss and accuracy metrics
        loss = logs['loss']
        accuracy = logs['accuracy']
        
        # Update the Streamlit interface with the current epoch's output
        st.text(f"Epoch {epoch}: loss = {loss:.4f}, accuracy = {accuracy:.4f}")


#run the app
if __name__ == "__main__":
    app()
