# %%
import logging
import sys

from llama_index import ServiceContext, SummaryIndex, VectorStoreIndex
from llama_index.llms import Ollama
from llama_index.readers import BeautifulSoupWebReader
from llama_index.response.notebook_utils import display_response

# %%
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


# %%
url_1 = (
    "https://www.theverge.com/"
    "2023/9/29/23895675/ai-bot-social-network-openai-meta-chatbots"
)
url_2 = (
    "https://raw.githubusercontent.com/run-llama/llama_index/main/examples/"
    "paul_graham_essay/data/paul_graham_essay.txt"
)

# %%
documents = BeautifulSoupWebReader().load_data([url_1, url_2])

# %%
llm = Ollama(model="llama2")

# %%
raw_llm_response = llm.complete("Who is Paul Graham?")
print(raw_llm_response)

# %%
# noinspection DuplicatedCode
service_context = ServiceContext.from_defaults(llm=llm)

# %%
vector_index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)

# %%
summary_index = SummaryIndex.from_documents(documents, service_context=service_context)

# %%
query_engine = vector_index.as_query_engine(respnse_mode="compact")

# %%
ai_response = query_engine.query(
    "What is the difference between OpenAI and " "Huggingface?"
)
print(ai_response)

# %%
pg_response = query_engine.query("Who is Paul Graham?")
print(pg_response)

# %%
response = query_engine.query("What did the author do growing up?")
print(response)

# %%
