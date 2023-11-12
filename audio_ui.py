import os

import gradio as gr
import numpy as np


def reverse_audio(audio):
    sr, data = audio
    return sr, np.flipud(data)


web_audio_files = ["https://samplelib.com/lib/preview/mp3/sample-3s.mp3"]
local_audio_files = [
    os.path.join(os.path.dirname(__file__), filename)
    for filename in ["audio/what-is-cam.wav"]
]
local_audio_files = [file for file in local_audio_files if os.path.exists(file)]

demo = gr.Interface(
    fn=reverse_audio,
    inputs=gr.Microphone(interactive=True, streaming=True),
    outputs=gr.Audio(streaming=True),
    examples=web_audio_files + local_audio_files,
    cache_examples=True,
)

if __name__ == "__main__":
    demo.launch()
