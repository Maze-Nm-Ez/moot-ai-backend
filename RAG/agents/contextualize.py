import os
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


class ContextualizeQuestion(BaseModel):
    """Contextualize the question."""
    contextualized_question: str = Field(...,
                                         description="The contextualized question.")


contextualize_q_system_prompt = (
    "You are an expert in rephrasing questions based *only* on a given chat history and a new user question. "
    "Your task is to formulate a standalone question that can be understood without the chat history. "
    "You MUST NOT use any external knowledge or make assumptions beyond what is explicitly stated in the chat history and the user question. "
    "For example, if 'Namal' is mentioned, and the chat history refers to Namal Rajapaksa the politician, you must assume 'Namal' is this person and not associate it with any university or other entity unless explicitly stated in the history."
    " Specifically:"
    "\n1. Replace pronouns (he, she, it, they, etc.) with the specific names or entities they refer to *from the chat history or question*."
    "\n2. Expand ambiguous references (e.g., 'that policy', 'his idea') to what they specifically refer to, using *only* information from the chat history or question."
    "\n3. Include relevant context from previous messages ONLY if it's necessary to understand the question and is present in the chat history."
    "\n4. Ensure the reformulated question is clear, self-contained, and ONLY uses information from the provided chat history and user question."
    "\n5. If a named entity (like a person's name, organization, or place) is mentioned, interpret it *solely* based on its usage within the chat history. Do not infer or associate it with any real-world entities or knowledge outside of the provided text."
    "\nDo NOT answer the question. Your ONLY job is to reformulate it based on the provided context."
)

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
structured_llm_router = llm.with_structured_output(ContextualizeQuestion)


contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

contextualizer = contextualize_q_prompt | structured_llm_router
