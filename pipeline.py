from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from elasticsearch import Elasticsearch
import os

model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
)

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

def compute_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    image_features = image_features / image_features.norm(
        dim=-1,
        keepdim=True
    )

    return image_features[0].cpu().numpy().tolist()

print(compute_image_embedding("./samples/skills-learning.png"))

def compute_prompt_embedding(prompt):
    inputs = processor(
        text=[prompt],
        return_tensors="pt",
        padding=True
    )

    text_features = model.get_text_features(**inputs)

    text_features = text_features / text_features.norm(
        dim=-1,
        keepdim=True
    )

    return text_features[0].detach().numpy()

print(compute_prompt_embedding("photo with a computer"))

class ELK:
    def __init__(self, address="http://localhost:9200"):
        self.address = address
        self.cluster = Elasticsearch(self.address)
        self.index = "cifp"
        self.body = {
        "mappings": {
            "properties": {
                "filepath": {
                    "type": "keyword"
                },
                "filename": {
                    "type": "text"
                },
                "embedding": {
                    "type": "dense_vector",
                    "dims": 512,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }
        self.setup()
        
    def setup(self):
        print("ELK/SETUP> Running...")
        if self._index_exist(self.index):
            print("ELK/SETUP> Already Setup!")
            return
        print("ELK/SETUP> Configuring...")
        try:
            self._index_create(self.index, self.body)
        except:
            raise "ELK/SETUP> Failed!"
            
        if self._index_exist(self.index):
            print("ELK/SETUP> Correctly Setup!")
            return
        raise "ELK/SETUP> Failed!"
        

    def _index_exist(self, name):
        return self.cluster.indices.exists(index=name)

    def _index_create(self, name, body):
        self.cluster.indices.create(index=name, body=body)
    
    def set_index(self, document):
        self.cluster.index(
            index=self.index,
            document=document
        )
        
    def query(self, query):
        return self.cluster.search(index=self.index, body=query)
        

elk_cluster = ELK()

def index_images(fpath):
    supported_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
    )

    for filename in os.listdir(fpath):

        if not filename.lower().endswith(
            supported_extensions
        ):
            continue

        filepath = os.path.join(
            fpath,
            filename
        )

        print(f"indexing: {filepath}")

        try:
            print("computing embedding...")
            embedding = compute_image_embedding(
                filepath
            )

            print("assembling document...")
            document = {
                "filepath": filepath,
                "filename": filename,
                "embedding": embedding
            }

            print("inserting into db...")
            print(f"preview document : {document}")
            elk_cluster.set_index(document)

            print(f"indexed: {filename}")

        except Exception as ex:
            print(f"err: {filename}")
            print(ex)
            
#index_images("./samples")

def search_images(prompt, k=10):
    print("computing prompt...")
    query_embedding = compute_prompt_embedding(prompt)
    resp = elk_cluster.query(
        {
            "size": k,
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": k,
                "num_candidates": 100
            }
        }
    )
    results = []

    for hit in resp["hits"]["hits"]:

        source = hit["_source"]

        results.append({
            "score": hit["_score"],
            "filepath": source["filepath"],
            "filename": source["filename"]
        })
    return results
    
print(search_images("with a computer", k=10))