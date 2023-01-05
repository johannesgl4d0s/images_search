from transformers import pipeline
from typing import List, Dict, Tuple
import numpy as np
from pathlib import Path
import pickle
import gc


class HuggingFaceImageClassifier:
    """
    HuggingFace Image Classifier, which uses ViT (Vision Transformer) as a base model.
    The model is trained on ImageNet dataset. Builds an index of images and finds similar images.

    Attributes:
        model_name (str): Name of the model to use. Default: google/vit-base-patch16-224
        index_file (str): Path to the index file. Default: ./data/index_hf.pickle
    """
    model_name = ""
    index_file = ""
    pipe = None
    index = None

    def __init__(self, model_name: str = "google/vit-base-patch16-224", 
                 index_file: str = "./data/index_hf.pickle"):
        self.model_name = model_name        
        self.index_file = index_file
        self.pipe = pipeline(task="image-classification", model=self.model_name)
        self.index = self.__load_index(self.index_file)
        
    def predict_image(self, image_path: str, top_k: int = 1000) -> List[Dict[float, str]]:
        """
        Predicts the image using the model.

        Args:
            image_path (str): Path to the image.
            top_k (int): Number of top predictions to return. Default: 1000

        Returns:
            List[Dict[float, str]]: List of top predictions.
        """
        return self.pipe(image_path, top_k=top_k)

    def __extract_features(self, image_path: str) -> Tuple[List[float], str]:
        """
        Extracts features and top class from the image.

        Args:
            image_path (str): Path to the image.

        Returns:
            Tuple[List[float], str]: Features and top class.
        """
        result = self.predict_image(image_path)
        top_class = result[0]["label"]
        features = sorted(result, key=lambda x: x["label"])
        features = [round(x["score"], 6) for x in features]
        return (features, top_class)

    def create_index(self, image_repo: str) -> None:
        """
        Creates an index of an images repository and saves pickle file.

        Args:
            image_repo (str): Path to the images repository.

        Returns:
            None    
        """
        images = list(Path(image_repo).iterdir())
        for i, image in enumerate(images):
            print(f"Processing {image.name}, Length of index {len(self.index)}")
            name = image.name
            features, top_class = self.__extract_features(image.resolve().__str__())
            self.index.append({"name": name, "top_class": top_class, "features": features})

            if i % 10 == 0 and i != 0: 
                print(f"Write index to {self.index_file}, Length: {len(self.index)}")                
                with open(self.index_file, "wb") as f:
                    pickle.dump(self.index, f, protocol=pickle.HIGHEST_PROTOCOL)
                gc.collect()            # garbage collection

    def __load_index(self, index_file: str) -> List[Dict]:
        """
        Loads index from pickle file.

        Args:
            index_file (str): Path to the index file.

        Returns:
            List[Dict]: List of images with features and top class.
        """
        if Path(index_file).exists() == False:
            print(f"Index file {index_file} not found. Please use create_index().")  
            return []
        with open(index_file, "rb") as f:
            index = pickle.load(f)
            return index

    def __cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """
        Calculates cosine similarity between two vectors.

        Args:
            a (List[float]): First vector.
            b (List[float]): Second vector.

        Returns:
            float: Cosine similarity.
        """
        a, b = np.array(a), np.array(b)
        return np.dot(a,b) / ( np.linalg.norm(a) * np.linalg.norm(b) )


    def find_similar_images(self, image_path: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Finds similar images in the index.

        Args:
            image_path (str): Path to the image.
            top_k (int): Number of top similar images to return. Default: 10

        Returns:
            List[Tuple[str, float]]: List of top similar images with cosine similarity score.
        """
        features, top_class = self.__extract_features(image_path)
        similar_images = list()

        for i, image in enumerate(self.index):
            if image["top_class"] == top_class:
                score = self.__cosine_similarity(features, image["features"])
                similar_images.append((image["name"], score))
        
        similar_images = sorted(similar_images, key=lambda x: x[1], reverse=True)
        return similar_images[:top_k]


if __name__ == "__main__":
    clf = HuggingFaceImageClassifier()
    clf.create_index("./img/imagenet-mini/")
    
    # Test
    uploaded_img = "./img/dog_input.jpg"
    similar_images = clf.find_similar_images(uploaded_img)
    print(similar_images)