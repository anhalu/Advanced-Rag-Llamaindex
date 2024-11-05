import os
from typing import Any, List 
from InstructorEmbedding import INSTRUCTOR 

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding

# Only for sentence-transformers==2.2.2 

class InstructorEmbeddings(BaseEmbedding): 
    _model: INSTRUCTOR = PrivateAttr()
    _instruction: str = PrivateAttr()
    
    def __init__(
        self, 
        instructor_model_name: str = 'dangvantuan/vietnamese-embedding', 
        cache_folder: str = '/home/hoang.minh.an/llm_weights/huggingface/hub', 
        instruction: str = "Represent a ducument for semantic search", 
        **kwargs: Any,
    ) -> None: 
        super().__init__(**kwargs) 
        self._model = INSTRUCTOR(instructor_model_name, cache_folder=cache_folder)
        self._instruction = instruction
        
    @classmethod 
    def class_name(cls) -> str:
        return 'instructor'
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        embeddings = self._model.encode([[self._instruction, query]])
        return embeddings[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings = self._model.encode([[self._instruction, text]])
        return embeddings[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(
            [[self._instruction, text] for text in texts]
        )
        return embeddings
