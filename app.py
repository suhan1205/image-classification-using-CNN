import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="CIFAR-10 Image Classifier", layout="centered")
st.title("🖼️ CIFAR-10 CNN Classifier")
st.write("Upload an image and the CNN will try to guess what it is!")

# 2. Load the Model (Cached so it only loads once)
@st.cache_resource
def load_cnn_model():
    # Make sure your model file is inside a folder named 'model'
    return tf.keras.models.load_model('/Users/sakshamkapoor/Downloads/ML-Summer_training/CNN_Streamlit/cifar10_cnn_model.h5')

with st.spinner("Loading Model..."):
    model = load_cnn_model()

# 3. Define the Classes (Extracted from your notebook)
CLASS_NAMES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# 4. Image Preprocessing Function
def preprocess_image(image):
    # Resize the image to 32x32 pixels as required by the model
    image = image.resize((32, 32))
    img_array = np.array(image)
    
    # Normalize the pixel values (0 to 1) just like in the notebook: X / 255.0
    img_array = img_array.astype('float32') / 255.0
    
    # Add a batch dimension: (32, 32, 3) becomes (1, 32, 32, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# 5. File Uploader UI
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # 6. Prediction Logic
    if st.button("Classify Image"):
        with st.spinner("Classifying..."):
            # Preprocess and predict
            processed_img = preprocess_image(image)
            predictions = model.predict(processed_img)
            
            # Extract the highest probability class
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]) * 100)
            predicted_label = CLASS_NAMES[predicted_class_idx]

        # Show Results
        st.success(f"**Prediction:** {predicted_label}")
        st.info(f"**Confidence:** {confidence:.2f}%")