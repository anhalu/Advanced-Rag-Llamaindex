import os
import torch
from typing import Optional, Any, List
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize



DEFAULT_HUGGINGFACE_LENGTH=512
DEFAULT_HUGGINGFACE_FOLDERS = os.getenv("HF_CACHE_DIR", None)
DEFAULT_EMBED_BATCH_SIZE=32


class CustomHuggingFaceEmbedding(BaseEmbedding):
    
    max_length: int = Field(
        default=DEFAULT_HUGGINGFACE_LENGTH, description="Maximum length of input.", gt=0
    )
    normalize: bool = Field(default=True, description="Normalize embeddings or not.")

    cache_folder: Optional[str] = Field(
        description="Cache folder for Hugging Face files.", default=None
    )
    
    _model: Any = PrivateAttr() 
    _device: str = PrivateAttr() 
    _parallel_process: bool = PrivateAttr()
    _target_device: Optional[List[str]] = PrivateAttr()
    
    def __init__(
        self, 
        model_name: str, 
        max_length: int = DEFAULT_HUGGINGFACE_LENGTH, 
        cache_folder: Optional[str] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE, 
        device: str = None, 
        normalize: bool = True,
        callback_manager: Optional[CallbackManager] = None , 
        parallel_process: bool = False,
        target_devices: Optional[List[str]] = None,
        language: str = 'vi', 
        **model_kwargs,
    ):
        device = device or 'cuda' if torch.cuda.is_available() else 'cpu'
        # device = 'cpu'
        cache_folder = cache_folder or DEFAULT_HUGGINGFACE_FOLDERS

        model = SentenceTransformer(
            model_name, 
            cache_folder=cache_folder, 
            device=device,
            **model_kwargs
        )
        
        if max_length: 
            model.max_seq_length = max_length
        else: 
            max_length = model.max_seq_length
            
        super().__init__(
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager, 
            model_name=model_name,
            max_length=max_length,
            normalize=normalize,
            device=device,
        )
        self._device = device
        self._model = model
        self._parallel_process = parallel_process
        self._target_device = target_devices
        self._language = language   
    
    @classmethod
    def class_name(cls) -> str:
        return "CustomHuggingFaceEmbedding"
    
    def _embed(
        self, 
        sentences: List[str], 
    ) -> List[List[str]]: 
        """Generates Embeddings either multiprocess or single process"""
        # if self._language == 'vi': 
        #     sentences = [tokenize(sent) for sent in sentences]
        
        if self._parallel_process: 
            pool = self._model.start_multi_process_pool(
                target_devices=self._target_devices
            )
            embs = self._model.encode_multi_process(
                sentences=sentences,
                pool=pool,
                batch_size=self.embed_batch_size,
                normalize_embeddings=self.normalize,
            )
            self._model.stop_multi_process_pool(pool=pool)
        else: 
            embs = self._model.encode(
                sentences,
                batch_size=self.embed_batch_size,
                normalize_embeddings=self.normalize,
            )
        return embs.tolist()
    
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed([query])[0]
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await self._embed([query])[0]
    
    
    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed([text])[0]
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await self._embed([text])[0]
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts) 
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return await self._embed(texts)
