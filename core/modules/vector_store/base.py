from typing import List, Any, Optional, Dict 
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.core.vector_stores import VectorStoreQuery, VectorStoreQueryResult
from llama_index.core.schema import TextNode, BaseNode


class BaseVectorStore(BasePydanticVectorStore): 
    
    store_text: bool = True
    
    def client(self) -> Any:
        """Get client."""
        pass
    
    def get(self, text_id: str) -> List[float]: 
        "Get the vector for a text_id"
        pass
    
    def add(
        self, 
        nodes: list[BaseNode], 
    ) -> List[str]: 
        "add nodes to index" 
        pass
    
    def delete(
        self, 
        ref_doc_id: str, 
        **delete_kwargs: Any
    ) -> None: 
        "delete nodes from index"
        pass
    
    def query(
        self, 
        query: VectorStoreQuery, 
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        "query the index"
        pass
    
    def persist(
        self, 
        persist_path: str, fs = None
    ) -> None: 
        "persist the index"
        pass
    