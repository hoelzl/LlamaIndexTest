# %%
import logging
import os.path
import sys

from llama_index import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)

# %%
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# %%
if not os.path.exists("./storage"):
    _documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(_documents)
    index.storage_context.persist()
else:
    _storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(_storage_context)

# %%
query_engine = index.as_query_engine()

# %%
response = query_engine.query("What did the author do growing up")
print(response)

# %%
missing_response = query_engine.query("When was the author born")
print(missing_response)
