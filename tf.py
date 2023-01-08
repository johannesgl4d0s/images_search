import keras
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
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
        self.index_file = index_file
        self.pca_file = pca_file
        print("Loading model...")
        self.model = keras.applications.VGG16(weights="imagenet", include_top=True)
        self.feature_extractor = Model(inputs=self.model.input, 
                                       outputs=self.model.get_layer("fc2").output)
        print("Loading index...")
        self.index = self.load_index(self.index_file)
        print("Loading PCA...")
        self.pca = self.load_pca(self.pca_file)

    def load_image(self, image_path: str) -> Tuple[Image.Image, np.ndarray]: 
        img = keras.utils.load_img(image_path, target_size=self.model.input_shape[1:3])
        x = keras.utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return img, x

    def extract_features(self, image_path: str) -> np.ndarray:
        img, x = self.load_image(image_path)
        features = self.feature_extractor.predict(x)
        return features

    def train_pca(self, features: np.ndarray) -> PCA:
        n = min(300, len(features))
        pca = PCA(n_components=n)
        pca.fit(features)
        with open("./data/pca_tf.pickle", "wb") as f:
            pickle.dump(pca, f)
        return pca

    def create_index(self, image_repo: str) -> None:
        features = []
        image_paths = []
        images = list(Path(image_repo).iterdir())
        for i, image_path in enumerate(images):
            print(f"Processing {image_path.name}, Length of index {len(features)}")
            feat = self.extract_features(image_path)[0]
            features.append(feat)
            image_paths.append(image_path)          # might have different order than images

        print("Extracting PCA features (might take some minutes)...")
        self.pca = self.train_pca(features)
        pca_features = self.pca.transform(features)
        self.index = [image_paths, pca_features]

        with open(self.index_file, "wb") as f:
            print(f"Saving index to {self.index_file}...")
            pickle.dump(self.index, f, protocol=pickle.HIGHEST_PROTOCOL)
            gc.collect()            # garbage collection
             
    def load_index(self, index_file: str) -> List:
        if Path(index_file).exists() == False:
            print(f"Index file {index_file} not found. Please use create_index().")  
            return None
        with open(index_file, "rb") as f:
            index = pickle.load(f)
            return index

    def load_pca(self, pca_file: str) -> PCA:
        if Path(pca_file).exists() == False:
            print(f"PCA file {pca_file} not found. Please use create_index().")  
            return None
        with open(pca_file, "rb") as f:
            pca = pickle.load(f)
            return pca

    def find_similar_images(self, image_path: str, top_k: int = 10) -> List[Tuple[str, float]]:
        new_features = self.extract_features(image_path)
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