from langchain_community.document_loaders import TextLoader
import os

file_names = ["namal.txt", "ranil.txt", "sajith.txt", "anura.txt"]
file_name_to_source = {
    "namal.txt": "namal",
    "ranil.txt": "ranil",
    "sajith.txt": "sajith",
    "anura.txt": "anura",
}
docs = []

for file_name in file_names:
    loader = TextLoader(f"documents/{file_name}", encoding="utf-8") 
    doc = loader.load()
    for d in doc:
        d.metadata = {"file_name": file_name, "source": file_name_to_source[file_name]}
        docs.append(d)

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# vectorstore = Chroma.from_documents(
#     documents=all_splits, embedding=OpenAIEmbeddings(), persist_directory="./vectore_stores/manifesto_vectorstore",
# )
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="./vectore_stores/manifesto_vectorstore")
# vectorstore.save_local("./vectore_stores/manifesto_vectorstore")

sajith_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6, "filter": {"source": "sajith"}})
namal_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6, "filter": {"source": "namal"}})
ranil_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6, "filter": {"source": "ranil"}})
anura_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6, "filter": {"source": "anura"}})