from RAG.agents.extractor import question_extractor


def extract_queries(state):
    """
    Extract queries for vector search and web search.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---EXTRACT QUERIES---")
    question = state["question"]
    source = question_extractor.invoke({"question": question})

    return {
        "namal_vector_search_query": source.namal_vector_search_query,
        "anura_vector_search_query": source.anura_vector_search_query,
        "web_search_query": source.web_search_query,
        "question": question
    }
