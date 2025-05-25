from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from typing import Literal

from RAG.graph_state import GraphState
from nodes import (
    load_context_and_create_query,
    retrieve_information,
    validate_retrieval_results,
    handle_retrieval_failure,
    generate_ai_response_with_context,
    validate_ai_output,
    handle_ai_generation_failure,
    comprehensive_evaluation_stage,
    aggregate_scores_and_flags,
    generate_structured_feedback,
    update_session_state,
    log_interaction,
    assemble_final_response,
    send_final_response,
)

workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("load_context_and_create_query",
                  load_context_and_create_query)
workflow.add_node("retrieve_information", retrieve_information)
workflow.add_node("validate_retrieval_results", validate_retrieval_results)
workflow.add_node("handle_retrieval_failure", handle_retrieval_failure)
workflow.add_node("generate_ai_response_with_context",
                  generate_ai_response_with_context)
workflow.add_node("validate_ai_output", validate_ai_output)
workflow.add_node("handle_ai_generation_failure", handle_ai_generation_failure)
workflow.add_node("comprehensive_evaluation_stage",
                  comprehensive_evaluation_stage)
workflow.add_node("aggregate_scores_and_flags", aggregate_scores_and_flags)
workflow.add_node("generate_structured_feedback", generate_structured_feedback)
workflow.add_node("update_session_state", update_session_state)
workflow.add_node("log_interaction", log_interaction)
workflow.add_node("assemble_final_response", assemble_final_response)
workflow.add_node("send_final_response", send_final_response)


# Define edges
workflow.add_edge(START, "load_context_and_create_query")
workflow.add_edge("load_context_and_create_query", "retrieve_information")
workflow.add_edge("retrieve_information", "validate_retrieval_results")

# Conditional edge for D_Decision


def decide_retrieval_sufficiency(state: GraphState) -> Literal["generate_ai_response_with_context", "handle_retrieval_failure"]:
    if state.get("is_retrieval_relevant_and_sufficient", False):
        return "generate_ai_response_with_context"
    else:
        return "handle_retrieval_failure"


workflow.add_conditional_edges(
    "validate_retrieval_results",
    decide_retrieval_sufficiency,
    {
        "generate_ai_response_with_context": "generate_ai_response_with_context",
        "handle_retrieval_failure": "handle_retrieval_failure",
    },
)

workflow.add_edge("handle_retrieval_failure", "assemble_final_response")
workflow.add_edge("generate_ai_response_with_context", "validate_ai_output")

# Conditional edge for F_Decision


def decide_ai_output_quality(state: GraphState) -> Literal["comprehensive_evaluation_stage", "handle_ai_generation_failure"]:
    if state.get("is_ai_output_valid", False):
        return "comprehensive_evaluation_stage"
    else:
        return "handle_ai_generation_failure"


workflow.add_conditional_edges(
    "validate_ai_output",
    decide_ai_output_quality,
    {
        "comprehensive_evaluation_stage": "comprehensive_evaluation_stage",
        "handle_ai_generation_failure": "handle_ai_generation_failure",
    },
)

# Diagram shows F_Fail --> G_Stage
workflow.add_edge("handle_ai_generation_failure",
                  "comprehensive_evaluation_stage")

# Comprehensive Evaluation connections
# G_Stage (comprehensive_evaluation_stage) is a single node in this implementation
# Its internal sub-evaluations (User, AI, Interaction) are handled within its own function.
# Then it directly goes to H (aggregate_scores_and_flags)

workflow.add_edge("comprehensive_evaluation_stage",
                  "aggregate_scores_and_flags")
workflow.add_edge("aggregate_scores_and_flags", "generate_structured_feedback")
workflow.add_edge("generate_structured_feedback", "update_session_state")
workflow.add_edge("update_session_state", "log_interaction")
workflow.add_edge("log_interaction", "assemble_final_response")
workflow.add_edge("assemble_final_response", "send_final_response")
workflow.add_edge("send_final_response", END)


memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# To generate a diagram (optional, requires extra dependencies like
# pip install pygraphviz Pillow, and having graphviz installed on your system)
try:
    graph_image = app.get_graph(xray=True).draw_mermaid_png()
    with open("new_graph_image.png", "wb") as f:
        f.write(graph_image)
    print("Graph image saved to new_graph_image.png")
except Exception as e:
    print(f"Could not generate graph image: {e}")

# Example of how to run (you'll need to provide initial state)
# config = {"configurable": {"thread_id": "some-thread-id"}}
# initial_input = {"user_input": "Hello, world!"} # Example input
# for event in app.stream(initial_input, config=config):
#     for k, v in event.items():
#         if k != "__end__":
#             print(v)
