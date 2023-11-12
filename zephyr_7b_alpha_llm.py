# %%
import logging
import sys

import torch
from llama_index import ServiceContext, SummaryIndex, VectorStoreIndex
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate
from llama_index.readers import BeautifulSoupWebReader
from llama_index.response.notebook_utils import display_response
from transformers import BitsAndBytesConfig

# %%
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


# %%
url = (
    "https://www.theverge.com/"
    "2023/9/29/23895675/ai-bot-social-network-openai-meta-chatbots"
)

# %%
documents = BeautifulSoupWebReader().load_data([url])

# %%
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


# %%
def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role in ["system", "user", "assistant"]:
            prompt += f"<|{message.role}|>\n{message.content}</s>\n"
        else:
            logging.warning(f"Unknown message role {message.role}. Skipping.")
    if not prompt.startswith("<|system|>\n"):
        prompt = f"<|system|>\n{prompt}"
    prompt = prompt + "<|assistant|>\n"
    return prompt


# %%
llm = HuggingFaceLLM(
    model_name="HuggingFaceH4/zephyr-7b-alpha",
    tokenizer_name="HuggingFaceH4/zephyr-7b-alpha",
    query_wrapper_prompt=PromptTemplate(
        "<|system|>\n</s><|user|>\n{query_str}</s>\n<|assistant|>\n"
    ),
    context_window=3900,
    max_new_tokens=256,
    model_kwargs={"quantization_config": quantization_config},
    generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    messages_to_prompt=messages_to_prompt,
    device_map="auto",
)

# %%
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
response = query_engine.query("What did the author do growing up?")
display_response(response)

# %%
