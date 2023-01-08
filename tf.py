import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import distance

from typing import Tuple, List
from PIL import Image
from pathlib import Path
import gc
import pickle

class KerasImageClassifier:
    model = None
    feature_extractor = None
    index_file = ""
    pca_file = ""
    index = None
    pca = None

    def __init__(self, 
                index_file: str = "./data/index_tf.pickle",
                pca_file: str = "./data/pca_tf.pickle"):
        """
        Keras Image Classifier using TensorFlow backend. 
        The model is trained on ImageNet dataset. Builds an index of images and finds similar images.

        Args:
            index_file (str, optional): Path to index file. Defaults to "./data/index_tf.pickle".
            pca_file (str, optional): Path to PCA file. Defaults to "./data/pca_tf.pickle".
        """
        self.index_file = index_file
        self.pca_file = pca_file
        print("Loading model...")
        self.model = keras.applications.VGG16(weights="imagenet", include_top=True)
        self.feature_extractor = Model(inputs=self.model.input, 
                                       outputs=self.model.get_layer("fc2").output)
        print("Loading index...")
        self.index = self.__load_index(self.index_file)
        print("Loading PCA...")
        self.pca = self.__load_pca(self.pca_file)

    def __load_image(self, image_path: str) -> Tuple[Image.Image, np.ndarray]: 
        """
        Preprocesses image for VGG16 model

        Args:
            image_path (str): Path to image
        
        Returns:
            Tuple[Image.Image, np.ndarray]: Image and preprocessed image
        """
        img = keras.utils.load_img(image_path, target_size=self.model.input_shape[1:3])
        x = keras.utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return img, x

    def __extract_features(self, image_path: str) -> np.ndarray:
        """
        Extracts features from image (VGG16 fc2 layer)

        Args:
            image_path (str): Path to image

        Returns:
            np.ndarray: Features
        """
        img, x = self.__load_image(image_path)
        features = self.feature_extractor.predict(x)
        return features

    def __train_pca(self, features: np.ndarray) -> PCA:
        """
        Train PCA on features (dimensionality reduction)

        Args:
            features (np.ndarray): Features

        Returns:
            PCA: PCA model
        """
        n = min(300, len(features))
        pca = PCA(n_components=n)
        pca.fit(features)
        with open(self.pca_file, "wb") as f:
            pickle.dump(pca, f)
        return pca

    def create_index(self, image_repo: str) -> None:
        """
        Creates index of images in image_repo and saves it to pickle file

        Args:
            image_repo (str): Path to image repository

        Returns:
            None
        """
        features = []
        image_paths = []
        images = list(Path(image_repo).iterdir())
        for i, image_path in enumerate(images):
            print(f"Processing {image_path.name}, Length of index {len(features)}")
            feat = self.__extract_features(image_path)[0]
            features.append(feat)
            image_paths.append(image_path)          # might have different order than images

        print("Extracting PCA features (might take some minutes)...")
        self.pca = self.__train_pca(features)
        pca_features = self.pca.transform(features)
        self.index = [image_paths, pca_features]

        with open(self.index_file, "wb") as f:
            print(f"Saving index to {self.index_file}...")
            pickle.dump(self.index, f, protocol=pickle.HIGHEST_PROTOCOL)
            gc.collect()            # garbage collection
             
    def __load_index(self, index_file: str) -> List:
        """
        Loads index from pickle file

        Args:
            index_file (str): Path to index file

        Returns:
            List: Index
        """
        if Path(index_file).exists() == False:
            print(f"Index file {index_file} not found. Please use create_index().")  
            return None
        with open(index_file, "rb") as f:
            index = pickle.load(f)
            return index

    def __load_pca(self, pca_file: str) -> PCA:
        """
        Loads PCA from pickle file

        Args:
            pca_file (str): Path to PCA file

        Returns:
            PCA: PCA model
        """
        if Path(pca_file).exists() == False:
            print(f"PCA file {pca_file} not found. Please use create_index().")  
            return None
        with open(pca_file, "rb") as f:
            pca = pickle.load(f)
            return pca

    def find_similar_images(self, image_path: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Finds similar images to image_path and returns top_k similar images, based on shortest distance

        Args:
            image_path (str): Path to image
            top_k (int, optional): Number of similar images to return. Defaults to 10.

        Returns:
            List[Tuple[str, float]]: List of tuples with similar image (image_path, distance)
        """
        new_features = self.__extract_features(image_path)
        new_pca_features = self.pca.transform(new_features)[0]
        distances = [ distance.cosine(new_pca_features, feat) for feat in self.index[1] ]
        idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[:top_k]
        similar_images = list(zip([self.index[0][i] for i in idx_closest],
                                  [distances[i] for i in idx_closest]))
        return similar_images


if __name__ == "__main__":
    clf = KerasImageClassifier()
    clf.create_index("./img/imagenet-mini/")

    # Test
    uploaded_img = "./img/dog_input.jpg"
    similar_images = clf.find_similar_images(uploaded_img)
    print(similar_images)