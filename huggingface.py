from transformers import pipeline
from typing import List, Dict, Tuple
import numpy as np
from pathlib import Path
import pickle
import gc


class HuggingFaceImageClassifier:
    model_name = ""
    index_file = ""
    pipe = None
    index = None

    def __init__(self, model_name: str = "google/vit-base-patch16-224", 
                 index_file: str = "./data/index_hf.pickle"):
        self.model_name = model_name        
        self.index_file = index_file
        self.pipe = pipeline(task="image-classification", model=self.model_name)
        self.index = self.load_index(self.index_file)
        
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
        data = list()
        for i, image in enumerate(images):
            print(f"Processing {image.name}")
            name = image.name
            features, top_class = self.extract_features(image.resolve().__str__())
            data.append({"name": name, "top_class": top_class, "features": features})

            if i % 3 == 0 and i != 0: 
                print(f"Write index to {self.index_file}")                
                index = self.load_index(self.index_file)
                index = index + data
                with open(self.index_file, "wb") as f:
                    pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
                index = list()
                data = list()
                gc.collect()            # garbage collection

    def load_index(self, index_file: str):
        index = []
        if Path(index_file).exists() == False:
            print(f"Index file {index_file} not found. Please use create_index().")  
        with open(index_file, "rb") as f:
            index = pickle.load(f)
        return index

    def cosine_similarity(a: List[float], b: List[float]):
        a, b = np.array(a), np.array(b)
        return np.dot(a,b) / ( np.linalg.norm(a) * np.linalg.norm(b) )


    def find_similar_images(self, image_path: str, top_k: int = 10) -> List[Tuple[str, float]]:
        features, top_class = self.extract_features(image_path)
        similar_images = list()

        for i, image in enumerate(self.index):
            if image["top_class"] == top_class:
                score = self.cosine_similarity(image["features"], features)
                similar_images.append((image["name"], score))
        
        similar_images = sorted(similar_images, key=lambda x: x[1], reverse=True)
        return similar_images[:top_k]


if __name__ == "__main__":
    clf = HuggingFaceImageClassifier()
    clf.create_index("./img/imagenet-mini/")

    #uploaded_img = "./img/dog_input.jpg"
    #similar_images = clf.find_similar_images(uploaded_img)
    #print(similar_images)

    #data = pickle.load(open("./data/index_hf.pickle", "rb"))
    #print(data)
    #print(data[0])
    #print(data[1].keys())