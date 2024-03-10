import os
from typing import List

from langchain_openai.embeddings import OpenAIEmbeddings
from llama_index.core import SimpleDirectoryReader
from llama_index.core.extractors import KeywordExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.llms.openai import OpenAI
from pinecone import Pinecone

from resources.playground_secret_key import PINECONE_KEY, SECRET_KEY

os.environ['PINECONE_API_KEY'] = PINECONE_KEY
environment = os.environ.get('PINECONE_ENVIRONMENT')
os.environ['OPENAI_API_KEY'] = SECRET_KEY


class Index:

    __index = Pinecone().Index('rag')
    __embed_model = OpenAIEmbeddings(model='text-embedding-ada-002')

    @classmethod
    def get_embed_model(cls) -> OpenAIEmbeddings:
        return cls.__embed_model

    @classmethod
    def __generate_metadata(cls, docs, keyword: bool = True):

        extractor = OpenAI(temperature=0.1, model="gpt-3.5-turbo", max_tokens=512)

        splitter = TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=128)

        transformations = [splitter]

        if keyword:
            transformations.append(KeywordExtractor(llm=extractor))

        # obtain file text (probably in a different way)

        ing_pipeline = IngestionPipeline(transformations=transformations)
        nodes = ing_pipeline.run(documents=docs)

        metadata = [node.metadata for node in nodes]
        text = [node.text for node in nodes]

        ids = [node.id_ for node in nodes]
        return metadata, text, ids  # need to decide how/where to get output

    @classmethod
    def populate(cls, filepaths: List[str]):
        vectors = []
        docs = SimpleDirectoryReader(input_files=filepaths).load_data()
        metadata, texts, ids = cls.__generate_metadata(docs)

        for j in range(len(metadata)):
            metadata[j]['text'] = texts[j]

        embeds = [cls.get_embed_model().embed_query(text) for text in texts]
        for i in range(len(texts)):
            vectors.append({
                'id': ids[i],
                'values': embeds[i],
                'metadata': metadata[i]


            })
        cls.__index.upsert(vectors=vectors, namespace='ns1')
        return

    @classmethod
    def add_file(cls):
        pass # TODO

    @classmethod
    def remove_file(cls):
        pass # TODO


if __name__ == '__main__':
    print(Index.populate(['../data/01_introduction_to_SL-4.pdf']))



