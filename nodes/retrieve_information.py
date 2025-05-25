from RAG.tools.vectore_store_retriever import document_retrievers


def retrieve_information(state):
    print("Node: Retrieve Information")
    query = state.get("query")
    # e.g., "namal" or "sc_fr_446_19-re-Jude-Samantha"
    target_source = state.get("target_source")

    if not query:
        print("No query provided to retrieve_information node.")
        state['retrieved_docs'] = []
        return state

    if target_source and target_source in document_retrievers:
        retriever = document_retrievers[target_source]
        print(f"Using retriever for source: {target_source}")
        retrieved_docs = retriever.invoke(query)
        state['retrieved_docs'] = retrieved_docs
    elif not target_source:
        print("No target_source specified. Consider iterating through all retrievers or using a general retrieval strategy.")
        # Placeholder: decide how to retrieve if no specific source is given
        # Maybe retrieve from all, or a default one.
        # For now, let's assume you might want to retrieve from ALL available sources if no specific one is mentioned
        all_retrieved_docs = []
        for source_name, retriever in document_retrievers.items():
            print(f"Retrieving from source: {source_name}")
            all_retrieved_docs.extend(retriever.invoke(query))
        state['retrieved_docs'] = all_retrieved_docs

    else:
        print(f"Retriever for source '{target_source}' not found.")
        state['retrieved_docs'] = []

    return state
