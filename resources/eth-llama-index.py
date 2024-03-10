import logging
import os
import sys

import chromadb
from eth_llama_config import embeddings_model_name, ollama_model_name
from llama_hub.file.unstructured import UnstructuredReader
from llama_hub.web.sitemap import SitemapReader
from llama_index import PromptHelper
from llama_index import ServiceContext
from llama_index import SimpleDirectoryReader
from llama_index import VectorStoreIndex
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import Ollama
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import ChromaVectorStore

print("You are about to re-index the Eurotech documentation. This will take a while, and will destroy the existing db. Are you sure you want to continue? (y/n)")

if input() != "y":
  sys.exit()

logging.basicConfig(stream=sys.stdout, level=logging.INFO) #DEBUG
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

os.environ["TOKENIZERS_PARALLELISM"]="true"


prompt_helper = PromptHelper(
  context_window=8192, # 8k context size Mistral-7B.Q8_0-Instruct-v0.2
  num_output=256,
  chunk_overlap_ratio=0.1,
  chunk_size_limit=None
)

llm = Ollama(model=ollama_model_name)  
embed_model = HuggingFaceEmbedding(model_name=embeddings_model_name)
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
    chunk_size=1024,
    chunk_overlap=128,
    prompt_helper=prompt_helper
)


indexData = False
indexESF = False
indexEC = False
indexSecurityOnly = False
indexManualsOnly = False

print("Index Full Data Directory? (y/n)")
if input() == "y":
 indexData = True
else:
  print("Index Security Manual? (y/n)")
  if input() == "y":
    indexSecurityOnly = True

  print("Index Manuals? (y/n)")
  if input() == "y":
    indexManualsOnly = True

print("Index ESF Documentation? (y/n)")
if input() == "y":
  indexESF = True

print("Index EC Documentation? (y/n)")
if input() == "y":
  indexEC = True

documents = [] 

if indexData:
  dir_reader = SimpleDirectoryReader(
      input_dir="../data",
      file_extractor={
        ".pdf": UnstructuredReader(),
        ".html": UnstructuredReader(),
      },
      recursive=True,
  )
  documents += dir_reader.load_data()

if indexSecurityOnly:
  documents += SimpleDirectoryReader('../data/security_manual').load_data()

if indexManualsOnly:
  documents += SimpleDirectoryReader('../data/manuals').load_data()

# web: EC Documentation
if indexEC:
  loader2 = SitemapReader()
  documents += loader2.load_data(sitemap_url='file:///home/demo/git/cx-demo-genai/src/sitemaps/ec.sitemap.xml', filter="https://ec.eurotech.com/docs/") #workaround for blocked sitemap

# web: ESF Documentation
if indexESF:
  loader3 = SitemapReader()
  documents += loader3.load_data(sitemap_url='file:///home/demo/git/cx-demo-genai/src/sitemaps/esf.sitemap.xml', filter="https://esf.eurotech.com/docs/") #workaround for blocked sitemap

# initialize client, setting path to save data
db = chromadb.PersistentClient(path="./chroma_db")

# create collection
chroma_collection = db.get_or_create_collection("iot")

# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents,
    service_context=service_context,
    storage_context=storage_context
)