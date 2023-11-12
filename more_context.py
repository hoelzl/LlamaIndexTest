# %%
import logging
import sys

from llama_index import VectorStoreIndex, SimpleDirectoryReader

# %%
documents = SimpleDirectoryReader("data").load_data()

# %%
index = VectorStoreIndex.from_documents(documents)

# %%
query_engine = index.as_query_engine(top_k=5)

# %%
response = query_engine.query("What did the author do growing up?")
print(response)
