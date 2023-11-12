import logging
import os
import sys

import gradio as gr
from llama_index import (
    BeautifulSoupWebReader,
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.llms import Ollama

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def cam(path):
    return f"https://coding-academy.com/en/{path}"


urls = [
    cam(path)
    for path in [
        "",
        "contact",
        "about-us",
        "programming-languages-machine-learning-open-training",
        "python-basic-course-for-beginners-and-non-programmers",
        "python-advanced-course-for-programmers-and-advanced",
        "python-clean-code-and-clean-test-for-programmers-and-advanced",
        "python-clean-software-architecture-and-design-patterns-"
        "for-programmers-and-advanced",
    ]
]

llm = Ollama(model="llama2")
service_context = ServiceContext.from_defaults(llm=llm, chunk_size=1000)

force_load = False
use_simple_data = False

if use_simple_data:
    if force_load or not os.path.exists("./storage"):
        _documents = SimpleDirectoryReader("data").load_data()
        vector_index = VectorStoreIndex.from_documents(_documents)
        vector_index.storage_context.persist()
    else:
        _storage_context = StorageContext.from_defaults(persist_dir="./storage")
        vector_index = load_index_from_storage(_storage_context)
else:
    if force_load or not os.path.exists("./storage"):
        _documents = BeautifulSoupWebReader().load_data(urls)
        vector_index = VectorStoreIndex.from_documents(_documents)
        vector_index.storage_context.persist()
    else:
        _storage_context = StorageContext.from_defaults(persist_dir="./storage")
        vector_index = load_index_from_storage(_storage_context)

use_chat_engine = True

if use_chat_engine:
    engine = vector_index.as_chat_engine(cache=None, similarity_top_k=6)
else:
    engine = vector_index.as_query_engine(cahce=None, similarity_top_k=6)


def response(message, _history):
    try:
        logging.info(f"Response for {message}")
        result = engine.chat(message) if use_chat_engine else engine.query(message)
        logging.info(f"Answer is {result}")
        return str(result)
    except:
        return "Sorry, an error occurred."


def random_response(message, _history):
    return "Random"


chat = gr.ChatInterface(
    response, title="Coding Academy Chatbot", chatbot=gr.Chatbot(height=650)
)

if __name__ == "__main__":
    chat.launch()
