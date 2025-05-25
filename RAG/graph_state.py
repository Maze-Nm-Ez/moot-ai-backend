from typing import List

from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages.base import BaseMessage


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    namal_vector_search_query: str
    ranil_vector_search_query: str
    sajith_vector_search_query: str
    anura_vector_search_query: str
    web_search_query: str
    generation: str

    namal_vector_search_documents: List[str]
    ranil_vector_search_documents: List[str]
    sajith_vector_search_documents: List[str]
    anura_vector_search_documents: List[str]
    web_search_documents: List[str]

    generated_count: int

    chat_history: Annotated[List[BaseMessage], add_messages]
    contextualized_question: str

    language: str
    translated_generation: str

    # New keys for conditional edges
    is_retrieval_relevant_and_sufficient: bool
    is_ai_output_valid: bool
    user_input: str  # Added for the first node
    context: dict  # Added for context loading
    query: str  # Added for the created query
    retrieved_docs: list  # For retrieved documents
    ai_response: str  # For AI generated response
    fallback_info: str  # For retrieval failure
    ai_error_fallback: str  # For AI generation failure
    # For comprehensive evaluation
    user_arg_score: float
    ai_response_score: float
    interaction_score: float
    aggregated_score: float
    flags: list
    structured_feedback: str
    final_response: str
