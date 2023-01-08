### Streamlit App for Similar Image Search Engine
"""
run with: 
$ python -m streamlit run app.py
"""
import streamlit as st
from PIL import Image


# Make Singleton Hugging Face model
@st.experimental_singleton
def get_hugging_face_model():
    from huggingface import HuggingFaceImageClassifier
    return HuggingFaceImageClassifier(index_file="./data/index_hf_25k.pickle")

# Make Singleton TensorFlow model
# (Caching not working due to pickle issues with TF)
@st.experimental_singleton
def get_tensorflow_model():
    from tf import KerasImageClassifier
    return KerasImageClassifier(index_file="./data/index_tf.pickle", pca_file="./data/pca_tf.pickle")


# Pre-Load both models
clf_tf = get_tensorflow_model()             
clf_hf = get_hugging_face_model()


# Sidebar
st.sidebar.markdown("# Similar Image Search Engine")
st.sidebar.markdown("Upload an image and find similar images in the database")
uploaded_img = st.sidebar.file_uploader("Upload an image", type=["jpg", "png"])

model_type = st.sidebar.radio("Model Type", ("Tensorflow", "Hugging Face"))

# Display uploaded image
st.markdown("## Uploaded Image")
if uploaded_img is not None:
    st.markdown("You have uploaded the following image:")
    img = Image.open(uploaded_img)
    st.image(img, width=300)

# Show similar images
st.markdown("## Similar Images")
st.markdown("The following images are similar to the uploaded image")

if uploaded_img is not None:
    with st.spinner("Calculating Scores/Distances..."):
        if model_type == "Tensorflow":
            similar_images = clf_tf.find_similar_images(img, top_k=10)
        else:
            similar_images = clf_hf.find_similar_images(img, top_k=10)

        for image, score in similar_images:        
            st.markdown(f"**{image}**")
            st.markdown(f"**Score/Distance: {score:.2f}**")
            st.image(f"./img/imagenet-mini/{image}", width=300)