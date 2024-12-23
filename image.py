import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pandas as pd
import pickle
import os

# Set page configuration
st.set_page_config(page_title="🖼️ Image Caption Generator", layout="wide")
st.title("🖼️ Image Caption Generator")

# Sidebar for additional details
st.sidebar.title("About the App")
st.sidebar.info(
    """
    This application uses a combination of **VGG16** and an **LSTM-based model** to generate captions for uploaded images.

    **How it works**:
    1. Upload an image in JPG, JPEG, or PNG format.
    2. The VGG16 model extracts features from the image.
    3. The LSTM model generates a caption based on the extracted features.

    Developed by: **Haris Faheem and Hassan Nasir**
    
    Contact: [LinkedIn Profile](https://www.linkedin.com/in/haris-faheem-1376982a3/)
        """
)

# Define the working directory for loading files
WORKING_DIR = '.'  # Current directory

# Load the trained model and associated files
@st.cache_resource
def load_model_and_assets():
    try:
        model = tf.keras.models.load_model(os.path.join(WORKING_DIR, 'best_model.keras'))
        with open(os.path.join(WORKING_DIR, 'tokenizer.pickle'), 'rb') as f:
            tokenizer = pickle.load(f)
        with open(os.path.join(WORKING_DIR, 'max_length.pkl'), 'rb') as f:
            max_length = pickle.load(f)
        return model, tokenizer, max_length
    except Exception as e:
        st.error(f"Error loading model or assets: {e}")
        return None, None, None

model, tokenizer, max_length = load_model_and_assets()

# Load VGG16 model for feature extraction
@st.cache_resource
def get_vgg16_model():
    vgg = VGG16()
    return tf.keras.Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)

vgg_model = get_vgg16_model()

# Helper function to convert index to word
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Function to predict image caption
def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    # Remove 'startseq' and 'endseq' from the generated caption
    final_caption = in_text.split()[1:-1]
    return ' '.join(final_caption)

# Initialize results storage
if 'results' not in st.session_state:
    st.session_state.results = []

def main():
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_image:
        st.markdown("---")
        st.subheader(f"Predictions for Uploaded Images")

        for image_file in uploaded_image:
            # Load and preprocess the uploaded image
            image = load_img(image_file, target_size=(224, 224))
            image_array = img_to_array(image)
            image_array = image_array.reshape((1, image_array.shape[0], image_array.shape[1], image_array.shape[2]))
            image_array = preprocess_input(image_array)

            # Feature extraction using VGG16
            feature = vgg_model.predict(image_array, verbose=0)

            # Predict caption
            with st.spinner(f"Generating caption for {image_file.name}..."):
                caption = predict_caption(model, feature, tokenizer, max_length)

            # Store the result
            st.session_state.results.append({"Image Name": image_file.name, "Caption": caption})

            # Display the image and predicted caption
            st.image(image_file, caption="Uploaded Image", use_column_width=True)
            st.success("Caption generated successfully!")
            st.write(f"**Generated Caption:** {caption}")

        # Display results as a DataFrame
        results_df = pd.DataFrame(st.session_state.results)
        st.markdown("## Results")
        st.dataframe(results_df)

        # Download button for CSV file
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Captions as CSV",
            data=csv,
            file_name="captions.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()
