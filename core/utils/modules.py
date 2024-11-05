from llama_index.core import Settings
from modules.models.vllm_llm import selfHostLLM
from modules.embed.huggingface_embedding import CustomHuggingFaceEmbedding
from .settings import Settings as AppSettings


def setup_modules(settings: AppSettings) -> dict:
    # self host llm model base on OpenAI completions service
    api_key = settings.openai.api_key
    base_url = settings.openai.base_url
    model = settings.openai.model

    llm = selfHostLLM(
        api_key=api_key, 
        base_url=base_url, 
        model=model
    )
    
    
    
    Settings.llm = llm

    