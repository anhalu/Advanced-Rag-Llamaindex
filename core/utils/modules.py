from llama_index.core import Settings
from core.modules.models.llms import selfHostLLM
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

    embed_model = CustomHuggingFaceEmbedding(
        model_name = settings.embed.embed_model, 
        max_length = settings.embed.max_length,
        cache_folder=settings.embed.cache_folder,
        embed_batch_size=settings.embed.embed_batch_size,
        language=settings.embed.language,
    )
    
    Settings.llm = llm
    Settings.embed_model = embed_model
    