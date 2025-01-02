from llama_index.core import SimpleDirectoryReader 
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.llms.azure_openai import AzureOpenAI
import os


OPENAI_ENDPOINT=os.getenv("OPENAI_ENDPOINT")    
OPENAI_KEY=os.getenv("OPENAI_KEY")
OPENAI_GPT_DEPLOYMENT_NAME=os.getenv("OPENAI_GPT_DEPLOYMENT_NAME")
OPENAI_API_VERSION=os.getenv("OPENAI_API_VERSION")
AZURE_OPENAI__EMBED_DEPLOYMENT_NAME=os.getenv("AZURE_OPENAI__EMBED_DEPLOYMENT_NAME")


client = AzureOpenAI(
    azure_endpoint=OPENAI_ENDPOINT,
    azure_deployment=OPENAI_GPT_DEPLOYMENT_NAME,
    api_key=OPENAI_KEY,
    api_version=OPENAI_API_VERSION,
    
)

with open("test/prompt_proposition.txt") as f:
    prompt = f.read()

content = "Python is a popular programming language. It is widely used in data science, artificial intelligence, and web development. Guido van Rossum created Python in 1991."

prompt = prompt.format(content=content)

response = client.complete(prompt=prompt, formatted=True)

print(response)