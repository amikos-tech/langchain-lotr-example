import os
from typing import Generator

from dotenv import load_dotenv

import chromadb
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain_community.document_transformers import (
    EmbeddingsClusteringFilter,
    EmbeddingsRedundantFilter,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

load_dotenv()
# # Get 3 diff embeddings.
all_mini = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
multi_qa_mini = HuggingFaceEmbeddings(model_name="multi-qa-MiniLM-L6-dot-v1")
filter_embeddings = OpenAIEmbeddings()
#
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(ABS_PATH, "db")
#
# # Instantiate 2 diff chromadb indexes, each one with a diff embedding.
client_settings = chromadb.config.Settings(
    is_persistent=True,
    persist_directory=DB_DIR,
    anonymized_telemetry=False,
)

#
# Define 2 diff retrievers with 2 diff embeddings and diff search type.


import csv

all_chromas = {}


def prepare_chromas():
    users = Chroma(
        collection_name="users",
        persist_directory=DB_DIR,
        client_settings=client_settings,
        embedding_function=all_mini,
    )
    messages = Chroma(
        collection_name="messages",
        persist_directory=DB_DIR,
        client_settings=client_settings,
        embedding_function=multi_qa_mini,
    )
    all_chromas["users"] = users
    all_chromas["messages"] = messages


def read_csv(file_path: str) -> Generator[list, None, None]:
    with open(file_path, 'r') as f:
        csvreader = csv.reader(f)
        next(csvreader)  # skip header
        yield from csvreader


def define_lotr():
    retriever_users = all_chromas["users"].as_retriever(
        search_type="similarity", search_kwargs={"k": 5, "include_metadata": True}
    )
    retriever_messages = all_chromas["messages"].as_retriever(
        search_type="mmr", search_kwargs={"k": 5, "include_metadata": True}
    )

    # The Lord of the Retrievers will hold the output of both retrievers and can be used as any other
    # retriever on different types of chains.
    lotr = MergerRetriever(retrievers=[retriever_users, retriever_messages])

    return lotr


if __name__ == "__main__":
    prepare_chromas()
    user_docs = []
    message_docs = []
    for row in read_csv('data.csv'):
        user_docs.append(row[0])
        message_docs.append(row[2])

    all_chromas["users"].add_documents([Document(page_content=doc) for doc in user_docs])
    all_chromas["messages"].add_documents([Document(page_content=doc) for doc in message_docs])
    lotr = define_lotr()

    # do something with lotr https://python.langchain.com/docs/integrations/retrievers/merger_retriever
