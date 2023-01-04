from transformers import pipeline
from typing import List, Dict
import numpy as np
from pathlib import Path
import json
import pickle
import gc


class HuggingFaceImagePredictor:
    model_name = "google/vit-base-patch16-224"
    index_path = "./data/index_hf.json"

    def __init__(self, model_name: str = None):
        if model_name is not None:
            self.model_name = model_name
        self.pipe = pipeline(task="image-classification", model=self.model_name)

        if Path(self.index_path).exists():
            self.index = self.load_index(self.index_path)
        else:
            print(f"Index file {self.index_path} not found. Please use create_index().")

    def predict_image(self, image_path: str) -> List[Dict[float, str]]:
        result = self.pipe(image_path, top_k=1000)
        result = sorted(result, key=lambda x: x["label"])
        return result

    def extract_features(self, image_path: str) -> List[float]:
        result = self.predict_image(image_path)
        return [x["score"] for x in result]

    def create_index(self, image_repo: str) -> None:
        images = list(Path(image_repo).iterdir())
        index = dict()
        for i, image in enumerate(images):
            print(f"Processing {image.name}")
            name = image.name
            features = self.extract_features(image.resolve().__str__())
            index.update({"name": name, "features": features})

            if i % 1000 == 0: 
                gc.collect()            # garbage collection
                break

        with open(self.index_path, "w", encoding="utf-8") as f:
            print(f"Write index to {self.index_path}")
            json.dump(index, f, ensure_ascii=False, indent=4)
        self.index = index

    def load_index(self, index_path: str):
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        return index

    def cosine_similarity():
        pass


if __name__ == "__main__":
    clf = HuggingFaceImagePredictor()
    clf.create_index("./img/imagenet")