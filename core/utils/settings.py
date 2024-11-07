from pydantic import BaseModel
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class TracingConfig(BaseModel): 
    trace: bool = True 
    public_key: str 
    secret_key: str 
    user_id: str 
    host: str = "https://langfuse.poc.sun-asterisk.ai"
    
    
class SelfHostLLM(BaseModel): 
    api_key: str 
    base_url: str 
    model: str


class EmbedConfig(BaseModel):
    embed_model: str
    max_length: int = 512
    normalize: bool = True
    cache_folder: Optional[str] = None
    embed_batch_size: int = 32
    language: str = 'vi'


class Settings(BaseSettings): 
    model_config = SettingsConfigDict(env_nested_delimiter="__")
    openai: SelfHostLLM
    embed: EmbedConfig
    tracing: Optional[TracingConfig]


def load_settings() -> Settings: 
    return Settings()
