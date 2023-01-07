import streamlit as st
from PIL import Image
import keras
import keras.utils as image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from glob import glob
from sklearn.decomposition import PCA
from scipy.spatial import distance
import random

model = keras.applications.VGG16(weights='imagenet', include_top=True)
feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
cat_files = glob('../test_set/cats/*.jpg')
dog_files = glob('../test_set/dogs/*.jpg')

images = cat_files

with open('test.npy', 'rb') as f:
    a = np.load(f)

features = np.array(a)
pca = PCA(n_components=300)
pca.fit(features)

pca_features = pca.transform(features)

def get_closest_images(query_image_idx, num_results=5):
    distances = [ distance.cosine(pca_features[query_image_idx], feat) for feat in pca_features ]
    idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[1:num_results+1]
    return idx_closest

def get_concatenated_images(indexes, thumb_height):
    thumbs = []
    for idx in indexes:
        img = image.load_img(images[idx])
        img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
        thumbs.append(img)
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
    return concat_image
def load_image(path):
    img = image.load_img(path, target_size=model.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x

def get_new_image(i):
    # load image and extract features
    new_image, x = load_image(i)
    new_features = feat_extractor.predict(x)

    # project it into pca space
    new_pca_features = pca.transform(new_features)[0]

    # calculate its distance to all the other images pca feature vectors
    distances = [ distance.cosine(new_pca_features, feat) for feat in pca_features ]
    idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[0:5]  # grab first 5
    results_image = get_concatenated_images(idx_closest, 200)
    return new_image,results_image



st.set_page_config(layout="wide", page_title="Image Background Remover")

st.write("## Remove background from your image")
st.write(
    ":dog: Try uploading an image to watch the background magically removed. Full quality images can be downloaded from the sidebar. This code is open source and available [here](https://github.com/tyler-simons/BackgroundRemoval) on GitHub. Special thanks to the [rembg library](https://github.com/danielgatis/rembg) :grin:"
)
st.sidebar.write("## Upload and download :gear:")

pca = PCA(n_components=300)

def fix_image(upload):
    i = Image.open(upload)
    new_image, results_image = get_new_image(i)

    col1.write("Original Image :camera:")
    col1.image(new_image)
    col1.image(i)

    col2.write("Fixed Image :wrench:")
    col2.image(results_image)


col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    fix_image(upload=my_upload)
else:
    fix_image("./dog_input.jpg")
#%%