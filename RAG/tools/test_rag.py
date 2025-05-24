from vectore_store_retriever import namal_retriever, anura_retriever
import sys
import os

# Add the project root to sys.path to allow for correct module resolution
project_root = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def test_retriever(retriever, query, name):
    print(f"\n--- Top results for {name} ---")
    results = retriever.get_relevant_documents(query)
    for i, doc in enumerate(results, 1):
        print(f"Result {i}:")
        print(doc.page_content)
        print(f"Metadata: {doc.metadata}")
        print("-" * 40)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cli/test_rag.py \"<your query>\"")
        sys.exit(1)
    query = sys.argv[1]

    test_retriever(namal_retriever, query, "Namal")
    test_retriever(anura_retriever, query, "Anura")
