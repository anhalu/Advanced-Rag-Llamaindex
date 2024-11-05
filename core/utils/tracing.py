from langfuse.llama_index import LlamaIndexCallbackHandler
from llama_index.core.callbacks import CallbackManager
from llama_index.core import Settings

from .settings import Settings as AppSettings


def setup_tracing(settings: AppSettings):
    trace = settings.tracing.trace
    if trace:
        callback_handler = LlamaIndexCallbackHandler(
            public_key=settings.tracing.public_key,
            secret_key=settings.tracing.secret_key,
            host=settings.tracing.host,
            user_id=settings.tracing.user_id
        )
        Settings.callback_manager = CallbackManager([callback_handler])