# %%
import logging
import os
import sys
from tempfile import NamedTemporaryFile

import gradio as gr
import numpy as np
import torch
from datasets import load_dataset
from elevenlabs import generate, play, set_api_key
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
# Use the elevenlabs API for text-to-speech
use_elevenlabs = True
# Use the bark model for text-to-speech
use_bark = True


# %%
def cam(path):
    return f"https://coding-academy.com/en/{path}"


# %%
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

# %%
llm = Ollama(model="llama2")
service_context = ServiceContext.from_defaults(llm=llm, chunk_size=1000)

# %%
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


# %%
# noinspection DuplicatedCode
if use_chat_engine:
    engine = vector_index.as_chat_engine(similarity_top_k=6)
else:
    engine = vector_index.as_query_engine(similarity_top_k=6)


# %%
def response(message):
    try:
        logging.info(f"Response for {message}")
        result = engine.chat(message) if use_chat_engine else engine.query(message)
        logging.info(f"Answer is {result}")
        return str(result)
    except:
        return "Sorry, an error occurred."


# %%
def transcribe_and_reply(recording):
    if recording is None:
        return "No recording", ""
    text = transcribe(recording)
    answer = response(text)
    return text, answer


# %%
if use_elevenlabs:
    set_api_key(os.environ["ELEVENLABS_API_KEY"])
else:
    if use_bark:
        synthesizer = pipeline(
            "text-to-speech",
            model="suno/bark-small",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        synthesizer = pipeline(
            "text-to-speech",
            model="microsoft/speecht5_tts",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        embeddings_dataset = load_dataset(
            "Matthijs/cmu-arctic-xvectors", split="validation"
        )
        speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(
            0
        )


# %%
def speak(text):
    logging.info(f"Speaking {text}")
    if use_elevenlabs:
        speech = generate(text, voice="Emily", model="eleven_multilingual_v2")
        # play(speech)
    elif use_bark:
        speech = synthesizer(
            text, forward_params={"do_sample": True}, return_tensors="pt"
        )
    else:
        speech = synthesizer(
            text, forward_params={"speaker_embeddings": speaker_embedding}
        )
    return speech


# %%
# _sound = speak("Hello, world!")


# %%
def speak_and_clear_recording(text):
    sound = speak(text)
    if use_elevenlabs:
        logging.info(f"Sound: {len(sound)}, {type(sound)})")
        with open("audio/temp.mp3", "wb") as f:
            f.write(sound)
        return {
            _output: "audio/temp.mp3",
            _recording: None,
        }
    else:
        return {_output: (sound["sampling_rate"], sound["audio"]), _recording: None}


# %%
app = gr.Blocks()

# %%
with app:
    _recording = gr.Microphone()
    _text = gr.Textbox()
    _label = gr.Label()
    _output = gr.Audio(format="mp3", autoplay=True)

    _recording.change(transcribe, inputs=_recording, outputs={_text})
    _text.change(response, inputs=_text, outputs=[_label])
    _label.change(
        speak_and_clear_recording, inputs=_label, outputs={_output, _recording}
    )


# %%
if __name__ == "__main__":
    app.launch()
