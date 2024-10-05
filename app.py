import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import random
import matplotlib.pyplot as plt

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('model/traffic_sign_model.h5')
    return model

model = load_model()

# Load label names if available
# Assuming you have a dictionary mapping label indices to names
label_names = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    # ... add all 43 classes
}

# Function to preprocess the image for prediction
def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((32, 32))  # Resize to 32x32
    image = np.array(image) / 255.0  # Normalize
    image = image.reshape(1, 32, 32, 1)  # Reshape for the model
    return image

# Randomly select images from the dataset (e.g., X_test)
def get_random_images(X_test, y_test, num_images=10):
    indices = random.sample(range(len(X_test)), num_images)
    images = [X_test[i] for i in indices]
    labels = [y_test[i] for i in indices]
    return images, labels

# Simulate the dataset (in reality, you will load your actual dataset here)
X_test = np.random.randint(0, 255, (1000, 32, 32, 3))  # 1000 random images (32x32 RGB)
y_test = np.random.randint(0, 43, 1000)  # Random labels from 0 to 42 (43 classes)

st.title("Traffic Sign Detection for Self-Driving Cars 🚗🚦")

st.markdown("""
This app uses a pre-trained CNN model to detect and classify traffic signs. 
You can either upload your own image or select from the sample images below.
""")

# Sidebar for navigation
st.sidebar.header("Navigation")
options = st.sidebar.selectbox("Choose a page", ["Home", "About", "Model Performance"])

if options == "Home":
    st.header("Detect Traffic Signs")

    # Display random sample images from the dataset
    st.subheader("Sample Traffic Sign Images")

    # Get random images from the test dataset
    random_images, random_labels = get_random_images(X_test, y_test, num_images=10)

    # Display the images in a grid using matplotlib
    W_grid = 5
    L_grid = 2
    fig, axes = plt.subplots(L_grid, W_grid, figsize=(10, 5))
    axes = axes.ravel()

    for i in np.arange(0, W_grid * L_grid):
        axes[i].imshow(random_images[i])
        axes[i].set_title(f"Label: {label_names.get(random_labels[i], 'Unknown')}", fontsize=8)
        axes[i].axis('off')

    plt.subplots_adjust(hspace=0.5)
    st.pyplot(fig)

    # Select a random image for prediction
    selected_image = random_images[random.randint(0, len(random_images) - 1)]

    if selected_image is not None:
        st.write("You selected an image.")

        # Display the selected image
        image = Image.fromarray(selected_image.astype('uint8'))
        st.image(image, caption='Selected Image', use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Predict the traffic sign
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][predicted_class]

        st.write(f"**Predicted Traffic Sign:** {label_names.get(predicted_class, 'Unknown')}")
        st.write(f"**Confidence:** {confidence*100:.2f}%")

    # Option to upload your own image
    st.subheader("Or upload your own image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Preprocess the uploaded image
        processed_image = preprocess_image(image)

        # Predict the traffic sign
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][predicted_class]

        st.write(f"**Predicted Traffic Sign:** {label_names.get(predicted_class, 'Unknown')}")
        st.write(f"**Confidence:** {confidence*100:.2f}%")

elif options == "About":
    st.header("About This Project")
    st.markdown("""
    This app demonstrates the use of Convolutional Neural Networks (CNNs) for detecting and classifying traffic signs. The model has been trained on the [German Traffic Sign Recognition Benchmark (GTSRB)] dataset and is capable of predicting 43 different types of traffic signs.
    
    **Key Features:**
    - Real-time traffic sign classification
    - Upload or choose from sample images
    - Trained on over 34,000 images
    
    **Technologies Used:**
    - TensorFlow/Keras
    - Streamlit
    - Python
    """)

elif options == "Model Performance":
    st.header("Model Performance")

    st.subheader("Training History")
    st.markdown("Here you can include some training history or performance metrics.")
    # You can load and display training history, confusion matrix, etc.