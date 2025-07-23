import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(to right, #e0eafc, #cfdef3);
    padding: 2rem;
}

/* Hero Section */
.hero {
    text-align: center;
    margin-bottom: 3rem;
}
.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(to right, #ff512f, #dd2476);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    font-size: 1.1rem;
    color: #444;
    margin-top: 10px;
}

/* Steps Section */
.steps-container {
    background: #fff;
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.1);
    margin-bottom: 2rem;
}
.step {
    margin-bottom: 1rem;
}
.step-title {
    font-weight: 600;
    color: #333;
}
.step-desc {
    color: #555;
}

/* Upload Card */
.stFileUploader > div {
    border: 2px dashed #888;
    border-radius: 15px;
    padding: 2rem;
    background-color: #fefefe;
    box-shadow: 0 8px 20px rgba(0,0,0,0.05);
    transition: all 0.3s ease;
}
.stFileUploader > div:hover {
    border-color: #dd2476;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(to right, #667eea, #764ba2);
    border: none;
    border-radius: 50px;
    padding: 0.8rem 2rem;
    color: white;
    font-size: 1rem;
    font-weight: 600;
    transition: 0.3s ease-in-out;
}
.stButton > button:hover {
    background: linear-gradient(to right, #764ba2, #667eea);
    transform: scale(1.05);
}

/* Recommendations */
.stImage > img {
    border-radius: 10px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    transition: transform 0.3s ease;
}
.stImage > img:hover {
    transform: scale(1.02);
}
.stRecommendation {
    font-size: 0.95rem;
    font-weight: 600;
    text-align: center;
    margin-top: 6px;
    color: #222;
}
</style>
""", unsafe_allow_html=True)

# Loading feature list and filenames
try:
    feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
    filenames = pickle.load(open('filenames.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading embeddings or filenames: {e}")
    st.stop()

# Loading ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# App title
st.markdown('<div class="stTitle">Fashion Recommender System</div>', unsafe_allow_html=True)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Function to save uploaded file
def save_uploaded_file(uploaded_file):
    try:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path  # Return the saved file path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

# Function to extract features from an image
def feature_extraction(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        return normalized_result
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        return None

# Function to find nearest neighbors
def recommend(features, feature_list):
    try:
        neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
        neighbors.fit(feature_list)
        distances, indices = neighbors.kneighbors([features])
        return indices
    except Exception as e:
        st.error(f"Error in recommendation: {e}")
        return None

# File upload handling
st.markdown("""
<div class="hero">
    <h1>👗 Fashion Recommender</h1>
    <p>Find similar styles by uploading your favorite fashion image.</p>
</div>

<div class="steps-container">
    <div class="step">
        <p class="step-title">📤 Step 1: Upload an Image</p>
        <p class="step-desc">Choose a photo of clothing or accessories that inspire you.</p>
    </div>
    <div class="step">
        <p class="step-title">🤖 Step 2: We Extract Features</p>
        <p class="step-desc">Using deep learning and ResNet50, we understand the style you're looking for.</p>
    </div>
    <div class="step">
        <p class="step-title">🛍️ Step 3: Get Recommendations</p>
        <p class="step-desc">View top 5 similar fashion pieces that match your image.</p>
    </div>
</div>
""", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)

    if file_path:
        # Display uploaded image with controlled size
        st.markdown('<div class="stHeader">Uploaded Image</div>', unsafe_allow_html=True)
        st.image(Image.open(file_path), caption="Uploaded Image", width=300)  # Set a fixed width for the uploaded image

        # Progress bar for feature extraction
        with st.spinner("Extracting features..."):
            features = feature_extraction(file_path, model)
            st.success("Feature extraction complete!")

        if features is not None:
            # Progress bar for recommendation
            with st.spinner("Finding recommendations..."):
                indices = recommend(features, feature_list)
                st.success("Recommendations ready!")

            if indices is not None:
                # Display recommended images
                st.markdown('<div class="stHeader">Recommended Fashion Items</div>', unsafe_allow_html=True)
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.image(filenames[indices[0][0]], use_container_width=True)
                    st.markdown('<div class="stRecommendation">Recommendation 1</div>', unsafe_allow_html=True)
                with col2:
                    st.image(filenames[indices[0][1]], use_container_width=True)
                    st.markdown('<div class="stRecommendation">Recommendation 2</div>', unsafe_allow_html=True)
                with col3:
                    st.image(filenames[indices[0][2]], use_container_width=True)
                    st.markdown('<div class="stRecommendation">Recommendation 3</div>', unsafe_allow_html=True)
                with col4:
                    st.image(filenames[indices[0][3]], use_container_width=True)
                    st.markdown('<div class="stRecommendation">Recommendation 4</div>', unsafe_allow_html=True)
                with col5:
                    st.image(filenames[indices[0][4]], use_container_width=True)
                    st.markdown('<div class="stRecommendation">Recommendation 5</div>', unsafe_allow_html=True)
        else:
            st.error("Feature extraction failed.")
    else:
        st.error("File upload failed.")