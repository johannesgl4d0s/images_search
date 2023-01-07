import keras
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import distance

from typing import Tuple
from PIL import Image
from pathlib import Path
import gc

class KerasImageClassifier:
    model = None
    feature_extractor = None
    index_file = ""
    index = None

    def __init__(self, index_file: str = "./data/index_tf.npy"):
        self.index_file = index_file
        self.model = keras.applications.VGG16(weights="imagenet", include_top=True)
        self.feature_extractor = Model(inputs=self.model.input, 
                                       outputs=self.model.get_layer("fc2").output)
        self.index = self.__load_index(self.index_file)


    def load_image(self, image_path: str) -> Tuple[Image.Image, np.ndarray]: 
        img = keras.utils.load_img(image_path, target_size=self.model.input_shape[1:3])
        x = keras.utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return img, x

    def extract_features(self, image_path: str) -> np.ndarray:
        img, x = self.load_image(image_path)
        features = self.feature_extractor.predict(x)[0]
        return features

    def extract_pca_features(self, features: np.ndarray) -> np.ndarray:
        pca = PCA(n_components=300)
        pca.fit(features)
        pca_features = pca.transform(features)
        return pca_features

    def create_index(self, image_repo: str) -> None:
        features = []
        images = list(Path(image_repo).iterdir())
        for i, image_path in enumerate(images):
            print(f"Processing {image_path.name}, Length of index {len(features)}")
            feat = self.extract_features(image_path)
            features.append(feat)

        print("Extracting PCA features...")
        self.index = self.extract_pca_features(features)
        
        with open(self.index_file, "wb") as f:
            print(f"Saving index to {self.index_file}...")
            np.save(f, self.index)
            gc.collect()            # garbage collection
             
    def load_index(self, index_file: str) -> np.ndarray:
        if Path(index_file).exists() == False:
            print(f"Index file {index_file} not found. Please use create_index().")  
            return []
        with open(index_file, "rb") as f:
            index = np.load(f)
            return index


if __name__ == "__main__":
    clf = KerasImageClassifier()
    img, x = clf.load_image("./img/dog_input.jpg")
    features = clf.extract_features("./img/dog_input.jpg")
    print("done")