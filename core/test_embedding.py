import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import Settings
from modules.embed.huggingface_embedding import CustomHuggingFaceEmbedding
from utils import initalize
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Settings.llm = llm
# embed_model = HuggingFaceEmbedding(
#    model_name="dangvantuan/vietnamese-embedding", cache_folder='/home/hoang.minh.an/llm_weights/huggingface/hub',
#     # model_name="vietgpt/vietnamese-sbert",  # <-- This is working perfectly
# )
# embed_model_name = os.getenv("EMBED_MODEL_NAME")

initalize() 

# documents = SimpleDirectoryReader(input_dir='./data').load_data()

# # embed_model = CustomHuggingFaceEmbedding(model_name=embed_model_name) 

# Settings.embed_model = embed_model
# Settings.chunk_size = 512

# index = VectorStoreIndex.from_documents(documents=documents)
# print(index)
# # response = index.as_query_engine().query(
# #     "Toán là gì ?"
# # )

# # print(response) 

