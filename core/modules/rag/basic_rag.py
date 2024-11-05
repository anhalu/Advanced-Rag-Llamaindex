from llama_index.core import SimpleDirectoryReader, VectorStoreIndex


data_path = 'data/ML_math.pdf'  
documents = SimpleDirectoryReader(input_dir='./data').load_data() 

