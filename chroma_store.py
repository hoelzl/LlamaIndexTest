# %%
import logging
import sys

import chromadb
from chromadb.db.base import UniqueConstraintError
from llama_index import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.vector_stores import ChromaVectorStore

# %%
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# %%
chroma_client = chromadb.PersistentClient()

# %%
try:
    chroma_collection = chroma_client.create_collection("quickstart")
except UniqueConstraintError:
    chroma_collection = chroma_client.get_collection("quickstart")

# %%
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# %%
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# %%
documents = SimpleDirectoryReader("data").load_data()

# %%
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# %%
query_engine = index.as_query_engine()

# %%
response = query_engine.query("What did the author do growing up?")
print(response)
