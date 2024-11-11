from .base import BaseVectorStore
import numpy as np
from llama_index.core.bridge.pydantic import Field 
from typing import List, Any, Optional, Dict, cast, Tuple
from llama_index.core.vector_stores import VectorStoreQuery, VectorStoreQueryResult
from llama_index.core.schema import  BaseNode
from llama_index.core.vector_stores import MetadataFilters

class VectorStore2(BaseVectorStore): 
    """Just Simple implementation of VectorStore""" 
    
    stores_text: bool = True
    node_dict: Dict[str, BaseNode] = Field(default_factory=dict)
    
    def get(self, text_id: str) -> List[float]: 
        "get embeddings" 
        return self.node_dict[text_id].embedding
    
    def add(self, nodes: List[BaseNode]) -> List[str]: 
        "add nodes to index" 
        for node in nodes: 
            self.node_dict[node.node_id] = node
    
    def delete(self, node_id: str, **delete_kwargs: Any) -> None:
        del self.node_dict[node_id]

def get_top_k_embeddings(
    query_embedding: List[float],
    doc_embeddings: List[List[float]],
    doc_ids: List[str],
    similarity_top_k: int = 5,
) -> Tuple[List[float], List]:
    """
        Get top nodes by similarity to the query.
        Semantic search
    """
    # dimensions: D
    qembed_np = np.array(query_embedding)
    # dimensions: N x D
    dembed_np = np.array(doc_embeddings)
    # dimensions: N
    dproduct_arr = np.dot(dembed_np, qembed_np)
    # dimensions: N
    norm_arr = np.linalg.norm(qembed_np) * np.linalg.norm(
        dembed_np, axis=1, keepdims=False
    )
    # dimensions: N
    cos_sim_arr = dproduct_arr / norm_arr

    # now we have the N cosine similarities for each document
    # sort by top k cosine similarity, and return ids
    tups = [(cos_sim_arr[i], doc_ids[i]) for i in range(len(doc_ids))]
    sorted_tups = sorted(tups, key=lambda t: t[0], reverse=True)

    sorted_tups = sorted_tups[:similarity_top_k]

    result_similarities = [s for s, _ in sorted_tups]
    result_ids = [n for _, n in sorted_tups]
    return result_similarities, result_ids
     
class VectorStore3A(VectorStore2): 
    """Implemens semantics/dense search""" 
    
    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Get nodes for response."""

        query_embedding = cast(List[float], query.query_embedding)
        doc_embeddings = [n.embedding for n in self.node_dict.values()]
        doc_ids = [n.node_id for n in self.node_dict.values()]

        similarities, node_ids = get_top_k_embeddings(
            query_embedding,
            doc_embeddings,
            doc_ids,
            similarity_top_k=query.similarity_top_k,
        )
        result_nodes = [self.node_dict[node_id] for node_id in node_ids]

        return VectorStoreQueryResult(
            nodes=result_nodes, similarities=similarities, ids=node_ids
        )
        
"""
    Add filter metadata. First we will first filter the candidate set with
    documents that pass the metatdata filter, and then perform semantic querying. 
"""

def filter_nodes(nodes: List[BaseNode], filters: MetadataFilters):
    filtered_nodes = []
    for node in nodes:
        matches = True
        for f in filters.filters:
            if f.key not in node.metadata:
                matches = False
                continue
            if f.value != node.metadata[f.key]:
                matches = False
                continue
        if matches:
            filtered_nodes.append(node)
    return filtered_nodes

# add filter nodes as a first-pass over the nodes beforew runing semanctic search 
def dense_search(query: VectorStoreQuery, nodes: List[BaseNode]): 
    query_embedding = cast(List[float], query.query_embedding) 
    doc_embeddings = [n.embedding for n in nodes]
    doc_ids = [n.node_id for n in nodes]
    
    return get_top_k_embeddings(
        query_embedding,
        doc_embeddings,
        doc_ids,
        similarity_top_k=query.similarity_top_k,
    )

class VectorStore3B(VectorStore2): 
    """Filter node -> Run Semantics Search"""
    
    def query(
        self, 
        query: VectorStoreQuery, 
        **kwargs: Any,
    ) -> VectorStoreQueryResult: 
        # 1. Use filter metadata to filter nodes
        nodes = self.node_dict.values() 
        
        if query.filters: 
            nodes = filter_nodes(nodes, query.filters)
        
        results_nodes = [] 
        similarities = [] 
        nodes_ids = [] 
        
        if len(nodes) > 0: 
            similarities, node_ids = dense_search(query, nodes) 
            result_nodes = [self.node_dict[node_id] for node_id in node_ids]
        return VectorStoreQueryResult(
            nodes=result_nodes, similarities=similarities, ids=node_ids
        )
        
        