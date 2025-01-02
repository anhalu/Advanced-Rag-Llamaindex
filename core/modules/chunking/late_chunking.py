from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_model = SentenceTransformer('dangvantuan/vietnamese-embedding', trust_remote_code=True).to(device)
transformer_layer = embedding_model._first_module()
pooling_layer = embedding_model._last_module()

text = "Berlin is the capital and largest city of Germany, both by area and by population. Its more than 3,85 million inhabitants make it the European Union's most populous city, as measured by population within city limits. The city is also one of the states of Germany, and is the third smallest state in the country in terms of area."
list_chunks = [chunk + '.' for chunk in text.split('.') if chunk]

# Step 1: Tokenize the entire text
tokens = embedding_model.tokenizer(text, return_tensors='pt', padding=False, truncation=False).to(device)


# Step 2: Get token embeddings
with torch.no_grad():
    outputs = transformer_layer({'input_ids': tokens['input_ids'], 'attention_mask': tokens['attention_mask']})
    token_embeddings = outputs['token_embeddings']

# Step 3: Use pooling layer for chunks
sentence_embeddings = []
current_token_idx = 1  # skip CLS token

for chunk in list_chunks:
    chunk_tokens = embedding_model.tokenizer(chunk, return_tensors='pt', padding=True, truncation=True).to(device)
    chunk_length = chunk_tokens['input_ids'].shape[1] - 2  # Remove CLS and SEP tokens
    
    chunk_embeddings = token_embeddings[:, current_token_idx:current_token_idx+chunk_length]
    chunk_attention_mask = chunk_tokens['attention_mask'][:, 1: -1] # Remove CLS and SEP tokens
    
    sentence_embedding = torch.mean(chunk_embeddings, dim=1)  # Mean pooling
    sentence_embedding = sentence_embedding.squeeze(0)  # Remove batch dimension
    print(sentence_embedding.shape)
    
    # # Use pooling layer
    # features = {}
    # features['token_embeddings'] = chunk_embeddings # Add batch dimension
    # features['attention_mask'] = chunk_attention_mask
    # features['sentence_embedding'] = torch.mean(chunk_embeddings, dim=1)  # Mean pooling
    # sentence_embedding = pooling_layer(features)['sentence_embedding']
    # sentence_embedding = sentence_embedding.squeeze(0)  # Remove batch dimension

    sentence_embeddings.append(sentence_embedding)
    current_token_idx += chunk_length

sentence_embeddings = torch.stack(sentence_embeddings)

# Step 4: Process query using pooling layer
query = "Berlin"
query_tokens = embedding_model.tokenizer(query, return_tensors='pt', padding=True, truncation=True).to(device)

with torch.no_grad():
    query_outputs = transformer_layer({'input_ids': query_tokens['input_ids'], 
                                     'attention_mask': query_tokens['attention_mask']})
    query_embedding = query_outputs['token_embeddings']

# use pooling layer
query_embedding = torch.mean(query_embedding, dim=1)  # Mean pooling
query_embedding = query_embedding.squeeze(0)  # Remove batch dimension


for sentence, embedding in zip(list_chunks, sentence_embeddings):
    print(f"Sentence: {sentence}")
    print(f"Cosine similarity: {cosine_similarity(embedding.cpu().numpy().reshape(1, -1), query_embedding.cpu().numpy().reshape(1, -1))}")