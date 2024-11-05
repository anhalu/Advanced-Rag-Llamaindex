from pydantic import BaseModel
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class TracingConfig(BaseModel): 
    trace: bool = True 
    public_key: str 
    secret_key: str 
    user_id: str 
    host: str = "https://langfuse.poc.sun-asterisk.ai"
    
    
class OpenAIConfig(BaseModel): 
    api_key: str 
    base_url: str 
    model: str


class Settings(BaseSettings): 
    model_config = SettingsConfigDict(env_nested_delimiter="__")
    tracing: Optional[TracingConfig]
    openai: OpenAIConfig
    

def load_settings() -> Settings: 
    return Settings()
