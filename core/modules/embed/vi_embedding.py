import os

from typing import Any
from sentence_transformers import SentenceTransformer 
from pyvi.ViTokenizer import tokenize 

Cache_dir = os.getenv('HF_CACHE_DIR', '/tmp')

class ViEmbedding:
    def __init__(self):
        self.model = SentenceTransformer('dangvantuan/vietnamese-embedding', cache_folder=Cache_dir)
    
    def get_embedding(self, text):
        return self.model.encode(self._tokenize(text))
    
    def _tokenize(self, text):
        return tokenize(text)
    
    def __call__(self, texts: list[str], *args: Any, **kwds: Any) -> Any:
        return [self.get_embedding(text) for text in texts]

if __name__ == "__main__": 
    vi_embedding = ViEmbedding()
    texts = ['Tôi là sinh viên trường đại học bách khoa hà nội']
    embeddings = vi_embedding(texts) 
    
    print(len(embeddings[0]))
    