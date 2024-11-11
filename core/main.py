from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from modules.embed.huggingface_embedding import CustomHuggingFaceEmbedding
from modules.vector_store.vector_store import VectorStore3B
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core import VectorStoreIndex
from utils import initalize

initalize()
text_path = '/home/hoang.minh.an/anhalu-data/learning/Advanced-Rag-Llamaindex/data/ML_math.pdf'
documents = SimpleDirectoryReader(input_files=[text_path]).load_data()

node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=20)
embed_model = CustomHuggingFaceEmbedding('dangvantuan/vietnamese-embedding', 
                                         max_length=256, 
                                         cache_folder='/home/hoang.minh.an/llm_weights/huggingface/hub')

nodes = node_parser.get_nodes_from_documents(documents=documents)

for node in nodes: 
    node_embedding = embed_model.get_text_embedding(node.get_content(metadata_mode='all'))
    node.embedding = node_embedding

vector_store = VectorStore3B() 
vector_store.add(nodes) 

query = "Hãy viết hàm norm với python." 
query_embedding = embed_model.get_query_embedding(query)

query_obj = VectorStoreQuery(
    query_embedding=query_embedding,
    similarity_top_k=2,
)

query_result = vector_store.query(query_obj)
for similarity, node in zip(query_result.similarities, query_result.nodes):
    print(
        "\n----------------\n"
        f"[Node ID {node.node_id}] Similarity: {similarity}\n\n"
        f"{node.get_content(metadata_mode='all')}"
        "\n----------------\n\n"
    )
    
    
# index = VectorStoreIndex.from_vector_store(vector_store)
# query_engine = index.as_query_engine() 

# query_str = "Hãy viết hàm norm với python" 

# response = query_engine.query(query_str)
# print(str(response))
