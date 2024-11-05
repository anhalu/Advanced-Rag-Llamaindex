from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Settings.llm = llm
embed_model = HuggingFaceEmbedding(
   model_name="dangvantuan/vietnamese-embedding", cache_folder='/home/hoang.minh.an/llm_weights/huggingface/hub',
    # model_name="vietgpt/vietnamese-sbert",  # <-- This is working perfectly
)

text = ['Hà Nội là thủ đô của Việt Nam', 'Đà Nẵng là thành phố du lịch']
embedding = embed_model.get_text_embedding_batch(text) 

print(len(embedding[0]))