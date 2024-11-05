from dotenv import load_dotenv 

from .settings import(
    Settings as AppSettings, 
    load_settings
)

from .tracing import setup_tracing
from .modules import setup_modules

def initalize(dotenv_path: str = '.env'):
    load_dotenv(dotenv_path)
    settings = load_settings()
    # setup_tracing(settings)
    setup_modules(settings)


__all__ = [
    'AppSettings',
    'load_settings',
    'setup_tracing',
    'setup_modules'
    'initalize'
]
