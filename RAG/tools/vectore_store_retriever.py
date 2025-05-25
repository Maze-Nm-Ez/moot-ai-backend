import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# Load environment variables from .env file
load_dotenv()
# Now you can access the API key
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")

DOCUMENTS_DIR = "documents/"
VECTORSTORE_DIR = "./vector_stores/main_document_vectorstore"

all_docs = []
sources = set()

print(f"Scanning directory: {DOCUMENTS_DIR}")
if not os.path.exists(DOCUMENTS_DIR):
    print(
        f"Warning: Documents directory '{DOCUMENTS_DIR}' not found. No documents will be processed.")
else:
    for filename in os.listdir(DOCUMENTS_DIR):
        file_path = os.path.join(DOCUMENTS_DIR, filename)
        # Get filename without extension as source
        source_name = os.path.splitext(filename)[0]

        if filename.lower().endswith(".txt"):
            print(f"Loading text file: {filename}")
            loader = TextLoader(file_path, encoding="utf-8")
            try:
                docs = loader.load()
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
        elif filename.lower().endswith(".pdf"):
            print(f"Loading PDF file: {filename}")
            loader = PyPDFLoader(file_path)
            try:
                docs = loader.load()
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
        else:
            print(f"Skipping unsupported file type: {filename}")
            continue

        for doc in docs:
            doc.metadata = {"source": source_name, "file_name": filename}
            all_docs.append(doc)
            sources.add(source_name)
        print(f"Successfully loaded and processed: {filename}")

document_retrievers = {}

if all_docs:
    print(
        f"Found {len(all_docs)} documents from {len(sources)} unique sources: {', '.join(sources)}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(all_docs)
    print(f"Split documents into {len(all_splits)} chunks.")

    print(f"Creating/updating vector store at: {VECTORSTORE_DIR}")
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=OpenAIEmbeddings(api_key=api_key),
        persist_directory=VECTORSTORE_DIR,
    )
    print("Vector store created/updated successfully.")

    print("Creating retrievers for each source...")
    for source_name in sources:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6, "filter": {"source": source_name}}
        )
        document_retrievers[source_name] = retriever
        print(f"Created retriever for source: {source_name}")
    print("All retrievers created.")
else:
    print("No documents found or processed. Vector store and retrievers will not be created.")

# Example of how to access a retriever:
# if "namal" in document_retrievers:
#     namal_retriever = document_retrievers["namal"]
#     # results = namal_retriever.invoke("some query")
# else:
#     print("Retriever for 'namal' not found.")

# To make retrievers available for import elsewhere:
# You can directly import `document_retrievers` dictionary from this module.
