"""
run with: 
$ python -m streamlit run app.py
"""
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# Basic Page Layout
st.markdown("# Similar Image Search Engine")
st.markdown("Upload an image and find similar images in the database")
uploaded_img = st.file_uploader("Upload an image", type=["jpg", "png"])

# Display uploaded image
if uploaded_img is not None:
    st.markdown("## Uploaded Image")
    img = Image.open(uploaded_img)
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.imshow(img)
    st.pyplot(fig)