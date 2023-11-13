# %%
import logging
import os
import sys

import gradio as gr
import numpy as np
from llama_index import (
    BeautifulSoupWebReader,
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.llms import Ollama
from transformers import pipeline

# %%
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# %%
# Possible values are "tiny", "base", "small", "medium", "large-v3"
model_size = "base"
# Forces the loading of the data into the vector database
force_load = False
# Use a simple local data set instead of scraping the web
use_simple_data = False
# Use the chat engine instead of the query engine
use_chat_engine = True


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

# noinspection DuplicatedCode
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


# %%
transcriber = pipeline(
    "automatic-speech-recognition", model=f"openai/whisper-{model_size}"
)


# %%
def transcribe(new_chunk):
    logging.info(f"Transcribing {new_chunk}")
    if new_chunk is None:
        return {_text: _text}
    sr, y = new_chunk
    y = y.astype(np.float32)
    y /= np.max(np.abs(y), initial=1.0)
    return {_text: transcriber({"sampling_rate": sr, "raw": y})["text"]}


# noinspection DuplicatedCode
if use_chat_engine:
    engine = vector_index.as_chat_engine(cache=None, similarity_top_k=6)
else:
    engine = vector_index.as_query_engine(cahce=None, similarity_top_k=6)


def response(message):
    try:
        logging.info(f"Response for {message}")
        result = engine.chat(message) if use_chat_engine else engine.query(message)
        logging.info(f"Answer is {result}")
        return str(result)
    except:
        return "Sorry, an error occurred."


def transcribe_and_reply(recording):
    if recording is None:
        return "No recording", ""
    text = transcribe(recording)
    answer = response(text)
    return text, answer


def clear_recording():
    return {_recording: None}


app = gr.Blocks()

with app:
    _recording = gr.Microphone()
    _text = gr.Textbox()
    _label = gr.Label()

    _recording.change(transcribe, inputs=_recording, outputs={_text})
    _text.change(response, inputs=_text, outputs=[_label])
    _label.change(clear_recording, outputs={_recording})


if __name__ == "__main__":
    app.launch()
