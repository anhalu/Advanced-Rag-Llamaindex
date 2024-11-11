from typing import Optional, List, Any, Sequence

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.llms import LLM
# from llama_index.llms.openai import OpenAI
from llama_index.core.base.llms.types import (
    LLMMetadata,
    ChatResponse,
    ChatMessage, CompletionResponse, ChatResponseGen, CompletionResponseGen, ChatResponseAsyncGen,
    CompletionResponseAsyncGen
)
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback
)

from openai import OpenAI 

class selfHostLLM(LLM): 
    api_key: str = Field(description="API key for Self host OpenAI.") 
    base_url: str = Field(description="Base URL for Self host OpenAI.") 
    
    model: str = Field(
        description="Model name for Self host OpenAI.", 
        default='Qwen/Qwen2.5-Coder-7B-Instruct-AWQ'
    )
    
    is_chat_model: bool = Field(
        default=False, 
        description="Whether the model is a chat model."
    )
    
    temperature: float = Field(
        default=0.5, 
        description="Temperature for sampling.", 
        ge=0,
        le=20,
    )
    
    max_output_tokens: Optional[int] = Field(
        description='Maximum number of tokens to generate.',
        default=None, 
    )
    
    _llm = PrivateAttr() 
    _chat_session = PrivateAttr()
    
    def __init__(
        self, 
        api_key: str, 
        base_url: str, 
        model: str = 'Qwen/Qwen2.5-Coder-7B-Instruct-AWQ',
        temperature: float = 0.5,
        max_output_tokens: Optional[int] = None,
        **kwargs
    ) -> None: 
        super().__init__(api_key=api_key,
                         base_url=base_url,
                         model=model,
                         temperature=temperature,
                         max_output_tokens=max_output_tokens,
                         # base class
                         **kwargs)
        self._llm = OpenAI(api_key=api_key, base_url=base_url)
        if self.is_chat_model: 
            self._chat_session = None
        
        self.model = model 
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        
    @classmethod
    def class_name(cls) -> str:
        return "SelfHostLLM"
    
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=32000,       # fix for qwen2.5 7B -> not use long context config
            num_output=8000, 
            is_chat_model=self.is_chat_model, 
            model_name=self.model
        )
        
    def _chat(self, messages: Sequence[ChatMessage], **kwargs) -> ChatResponse:
        # call llm 
        response = self._llm.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        chat_message = ChatMessage(
            role='assistant',
            content=response.choices[0].message.content,
        )
        return ChatResponse(
            message=chat_message,
        )
        
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        raise NotImplementedError("This class does not support streaming chat")

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return self._chat(messages, **kwargs)

    def _complete(self, prompt: str, **kwargs: Any) -> CompletionResponse: 
        response = self._llm.completions.create(
            model=self.model,
            prompt=prompt,
            # **kwargs
        )
        return CompletionResponse(
            text=response.choices[0].text
        )
    
    def _stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError("This class does not support streaming completion")
    
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        return self._stream_complete(prompt, **kwargs)

    @llm_completion_callback() 
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return self._complete(prompt, **kwargs)


    # ==== Async Endpoints ====
    @llm_chat_callback()
    def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        return self._chat(messages, **kwargs)

    @llm_chat_callback()
    def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        raise NotImplementedError("This class does not support streaming chat")

    @llm_completion_callback()
    def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return self._complete(prompt)

    @llm_completion_callback()
    def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        raise NotImplementedError("This class does not support streaming completion")
