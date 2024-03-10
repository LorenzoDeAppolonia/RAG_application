# From https://docs.llamaindex.ai/en/stable/understanding/storing/storing.html#using-vector-stores

import logging
import os
import sys

import chromadb
import gradio as gr
from chromadb.config import Settings
from conf.config import embeddings_model_name
from llama_index import PromptHelper
from llama_index import ServiceContext
from llama_index import VectorStoreIndex
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import ChatMessage
from llama_index.llms import MessageRole
from llama_index.llms import OpenAILike
from llama_index.prompts import ChatPromptTemplate
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import ChromaVectorStore


#
# internal functions
#

# define prompt viewing function
def display_prompt_dict(prompts_dict):
    for k, p in prompts_dict.items():
        text_md = k + "\n"
        logging.getLogger().debug(text_md)
        logging.getLogger().debug(p.get_template())
        logging.getLogger().debug("\n\n")

def data_querying(input_text, follow_up_questions = False):
  response = query_engine.query(input_text)
  return response


#
# main
#

# env and logger
os.environ["TOKENIZERS_PARALLELISM"]="true"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG) #DEBUG
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# system prompt
SYSTEM_PROMPT=(
    "You are an expert Q&A system, answering questions on Eurotech products. "
    "Never offend or attack or use bad words against Eurotech. "
    "Always answer the query using the provided context from the Eurotech documentation and not prior knowledge. "
    "If you don't know the answer, just say that you don't know, don't try to make up an answer. "
    "Some rules to follow:\n"
    "1. Provide specific answers in bullet points if you are returing a list or a sequence of steps\n"
    "2. Never directly reference the given context in your answer.\n"
    "3. Avoid statements like 'Based on the context, ...' or ' Based on the provided context information, ...' or 'The context information ...' or anything along those lines.\n"
)

# Text QA Prompt
chat_text_qa_msgs = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=(
            SYSTEM_PROMPT
        ),
    ),
    ChatMessage(
        role=MessageRole.USER,
        content=(
            "<s>[INST]"
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, answer the question: {query_str} \n"
            "[/INST]"
        ),
    ),
]
text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)

# Refine Prompt
chat_refine_msgs = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=(
            SYSTEM_PROMPT
        ),
    ),
    ChatMessage(
        role=MessageRole.USER,
        content=(
            "<s>[INST]"
            "The original query is as follows: {query_str}"
            "We have provided an existing answer: {existing_answer}"
            "We have the opportunity to refine the existing answer (only if needed) with some more context below."
            "------------"
            "{context_msg}"
            "------------"
            "Given the new context, refine the original answer to better answer the query. If the context isn't useful, return the original answer."
            "Refined Answer: "
            "[/INST]"
        ),
    ),
]
refine_template = ChatPromptTemplate(chat_refine_msgs)

prompt_helper = PromptHelper(
  context_window=32768, # 32k context size Mistral-7B.Q8_0-Instruct-v0.2
  num_output=256,
  chunk_overlap_ratio=0.1,
  chunk_size_limit=None
)

# llm

# LM Studio on macOS
llm = OpenAILike(api_base="http://localhost:1234/v1", 
                 api_key="nokey",
                 temperature=0.1)

# Ollama local
# llm = Ollama(
#     model = ollama_model_name,
#     temperature=0.1,
#     request_timeout=300.0 # This solves the time out error
# )

# embeddings
embed_model = HuggingFaceEmbedding(model_name=embeddings_model_name)

# service context
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
    chunk_size=1024,
    chunk_overlap=128,
    prompt_helper=prompt_helper
)

# storage
db = chromadb.PersistentClient(
  path="./chroma_db", 
  settings=Settings(anonymized_telemetry=False)
)
chroma_collection = db.get_or_create_collection("eth")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# index
index = VectorStoreIndex.from_vector_store(
    service_context=service_context,
    vector_store=vector_store
)

# query engine
query_engine = index.as_query_engine(
    service_context=service_context,
    text_qa_template=text_qa_template,
    refine_template=refine_template,
    similarity_top_k=4
)
query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": text_qa_template}
)
query_engine.update_prompts(
    {"response_synthesizer:refine_template": refine_template}
)

# debug prompts
prompts_dict = query_engine.get_prompts()
display_prompt_dict(prompts_dict)


# gradio interface 
iface = gr.ChatInterface(data_querying)
iface.launch(share=False)

# in a hardened configuration, what are the everyware linux users defined?
