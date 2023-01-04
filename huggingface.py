from transformers import pipeline
from typing import List, Dict, Tuple
import numpy as np
from pathlib import Path
import pickle
import gc


class HuggingFaceImagePredictor:
    model_name: str = "google/vit-base-patch16-224"
    index_file: str = "./data/index_hf.pickle"

    def __init__(self, model_name: str = None):
        if model_name is not None:
            self.model_name = model_name        
        self.pipe = pipeline(task="image-classification", model=self.model_name)

        if Path(self.index_file).exists():
            self.index = self.load_index(self.index_file)
        else:
            print(f"Index file {self.index_file} not found. Please use create_index().")  
        
    def predict_image(self, image_path: str, top_k: int = 1000) -> List[Dict[float, str]]:
        return self.pipe(image_path, top_k=top_k)

    def extract_features(self, image_path: str) -> Tuple[List[float], str]:
        result = self.predict_image(image_path)
        top_class = result[0]["label"]
        features = sorted(result, key=lambda x: x["label"])
        features = [round(x["score"], 6) for x in result]
        return (features, top_class)

    def create_index(self, image_repo: str) -> None:
        images = list(Path(image_repo).iterdir())
        index = dict()
        for i, image in enumerate(images):
            print(f"Processing {image.name}")
            name = image.name
            features, top_class = self.extract_features(image.resolve().__str__())
            index.update({"name": name, "top_class": top_class, "features": features})

            if i == 10: 
                gc.collect()            # garbage collection
                break

        with open(self.index_file, "w") as f:
            print(f"Write index to {self.index_file}")
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.index = index

    def load_index(self, index_file: str):
        with open(index_file, "r") as f:
            index = pickle.load(f)
        return index

    def cosine_similarity():
        pass


if __name__ == "__main__":
    clf = HuggingFaceImagePredictor()
    clf.create_index("./img/imagenet")