from langchain_openai.embeddings import OpenAIEmbeddings
import os
import sys
from pinecone import Pinecone
from resources.playground_secret_key import PINECONE_KEY_2, SECRET_KEY, PINECONE_KEY
from typing import List
from llama_index.llms.openai import OpenAI
from llama_index.core.extractors import KeywordExtractor, QuestionsAnsweredExtractor
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline


os.environ['PINECONE_API_KEY'] = PINECONE_KEY_2
environment = os.environ.get('PINECONE_ENVIRONMENT')
os.environ['OPENAI_API_KEY'] = SECRET_KEY


class Index:

    __index = Pinecone().Index('rag')
    __embed_model = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=3072)

    @classmethod
    def get_embed_model(cls) -> OpenAIEmbeddings:
        return cls.__embed_model

    @classmethod
    def __generate_metadata(cls, docs, keyword: bool = True, qna : bool = False):

        """Used to create the metadata associated to a chunk of text from documents.

        :param docs: documents inside a list (output of the directory reader).
        :param keyword: to specify whether to generate keywords from the chunk.
        :param qna: to specify whether to generate a series of question and their respective answers related to chunk.
        :return: metadata: list of dictionaries containing the metadata itself.
                 text: list containing the content of each node.
                 ids: list containing the IDs of the nodes generated by the extractor.

                Some possible metadata keys include:
                 * 'page_label': (str) label to identify the page.
                 * 'file_name': (str) name of the file. Last element of the path.
                 * 'file_path': (str) the local path.
                 * 'file_type': (str) MIME type of the file.
                 * 'file_size': (int) size of the file in bytes.
                 * 'creation_date': (str)
                 * 'last_modified_date': (str)
                 * 'document_title': (str) title automatically generated by the extractor.
                 * 'excerpt_keywords': (str, optional) keywords automatically generated by the extractor if specified.
                 * 'questions_this_excerpt_can_answer': (str, optional) questions and answers automatically generated by the extractor if specified.
                    Long process, thus suggested only with very important documents.
        """

        extractor = OpenAI(temperature=0.1, model="gpt-3.5-turbo", max_tokens=512)

        splitter = TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=128)

        transformations = [splitter]

        if keyword:
            transformations.append(KeywordExtractor(llm=extractor))

        if qna:
            transformations.append(QuestionsAnsweredExtractor(llm=extractor))

        # obtain file text (probably in a different way)

        ing_pipeline = IngestionPipeline(transformations=transformations)
        nodes = ing_pipeline.run(documents=docs)

        metadata = [node.metadata for node in nodes]
        text = [node.text for node in nodes]

        ids = [node.id_ for node in nodes]
        return metadata, text, ids  # need to decide how/where to get output

    @classmethod
    def __get_size_of(cls,obj):
        """
        Used to get the size of an object, generally an embedding vector.
        :param obj:
        :return: the size of the object.
        """

        size = sys.getsizeof(obj)
        if isinstance(obj, dict):
            size += sum([cls.__get_size_of(v) for v in obj.values()])
            size += sum([cls.__get_size_of(k) for k in obj.keys()])
        elif hasattr(obj, '__dict__'):
            size += cls.__get_size_of(obj.__dict__)
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
            size += sum([cls.__get_size_of(i) for i in obj])
        return size

    @classmethod
    def populate(cls, filepaths: List[str] = None, directory : str = None, keyword: bool = True, qna : bool = False, metadata : bool = True):
        """
        Used to populate the vector index with chunks of text and their metadata.
        A mechanism is used to check if the total vectors to upsert would exceed the 2MB allowed by Pinecone for a
        single request, in that case the request is split into many.

        :param filepaths: List of filepaths to target specific files.
        :param directory: Single directory to recursively target all of the files. (!!! CHECK IF THIS ACTUALLY WORKS !!!)
        :param keyword: Used to decide if the metadata generator will create keywords for the chunks.
        :param qna: Used to decide if the metadata generator will create question and answer pairs for the chunks.
        :return: None
        """

        required_exts = ['.pdf', '.csv', '.xlsx', 'docx']

        if directory is not None:
            docs = SimpleDirectoryReader(input_dir=directory, recursive=True, required_exts=required_exts).load_data()
            print('extracted docs with SimpleDirectoryReader')
        else:
            docs = SimpleDirectoryReader(input_files=filepaths).load_data()

        print('calling __generate_metadata method')
        metadata, texts, ids = cls.__generate_metadata(docs,keyword,qna)
        print('created metadata')

        for j in range(len(metadata)):
            metadata[j]['text'] = texts[j]

        embeds = [cls.get_embed_model().embed_query(text) for text in texts]

        cur_batch = []
        cur_batch_size = 0
        batch_size_limit = 2 * 1024 * 1024

        for i in range(len(texts)):
            vector_size = cls.__get_size_of(embeds[i])

            if cur_batch_size+vector_size>batch_size_limit:
                cls.__index.upsert(vectors=cur_batch, namespace='ns1')
                cur_batch = []
                cur_batch_size = 0

            cur_batch.append({
                'id': ids[i],
                'values': embeds[i],
                'metadata': metadata[i]})
            cur_batch_size += vector_size

        if cur_batch:
            cls.__index.upsert(vectors=cur_batch, namespace='ns1')
        return

    @classmethod
    def add_file(cls, filepath : str):
        pass # TODO

    @classmethod
    def remove_file(cls, filepath : str):
        pass
#         cls.__index.delete(namespace='ns1',
#     filter={
#         "file_path": {"$eq": filepath}
#     }
# )
        # starter index doesn't allow for delete with metadata :(


if __name__ == '__main__':
    print(Index.populate(directory='../data/00_materiale_di_partenza', keyword=False))


