from modules.models.vllm_llm import selfHostLLM
from llama_index.core.prompts import ChatMessage
# from utils import initalize 

# initalize()

llm = selfHostLLM(
    api_key='anhalu', 
    base_url='http://localhost:11434/v1', 
    model='Qwen/Qwen2.5-Coder-7B-Instruct-AWQ',
)
messages = [
    ChatMessage(
        role='system',content='You are a chatbot.',
    ), 
    ChatMessage(
        role='user', content='What is the weather today?',
    )
]
response = llm.chat(messages=messages)
print(response)
