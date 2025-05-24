from RAG.agents.compare_generate import rag_chain


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    web_documents = state["web_search_documents"]
    namal_vector_documents = state["namal_vector_search_documents"]
    anura_vector_documents = state["anura_vector_search_documents"]

    # RAG generation
    generation = rag_chain.invoke(
        {
            "web_context": web_documents,
            "namal_context": namal_vector_documents,
            "anura_context": anura_vector_documents,
            "question": question,
        }
    )

    generated_count = state.get("generated_count", 0) + 1
    return {
        "question": question,
        "generation": generation,
        "generated_count": generated_count,
    }
