import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

from utils import initalize

from llama_index.core import Settings
from llama_index.core import PromptTemplate
from llama_index.core.prompts import ChatMessage

initalize()
query_str = "How do the models developed in this work compare to open-source chat models based on the benchmarks tested?"
query_gen_prompt_str = (
    "You are a helpful assistant that generates multiple search queries based on a "
    "single input query. Generate {num_queries} search queries, one on each line, "
    "related to the following input query:\n"
    "Query: {query}\n"
    "Queries:\n"
)
query_gen_prompt = PromptTemplate(query_gen_prompt_str)
def generate_queries(llm, query_str: str, num_queries: int = 4):
    fmt_prompt = query_gen_prompt.format(
        num_queries=num_queries - 1, query=query_str
    )
    print(fmt_prompt)
    response = llm.complete(fmt_prompt)
    print(response)
    queries = response.text.split("\n")
    return queries

queries = generate_queries(Settings.llm, query_str, num_queries=4)

print(queries)