import os
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_openai import OpenAI


class ContextualizeQuestion(BaseModel):
    """Contextualize the question."""
    contextualized_question: str = Field(...,
                                         description="The contextualized question.")


contextualize_q_system_prompt = (
    "Given a chat history and the latest user question, "
    "which might reference context in the chat history, "
    "formulate a standalone question that can be understood "
    "without the chat history. Specifically:"
    "\n1. Replace pronouns with their specific referents."
    "\n2. Expand references to what they specifically refer to."
    "\n3. Include any relevant context from previous messages."
    "\n4. Ensure the reformulated question is clear and self-contained."
    "\nDo NOT answer the question, just reformulate it."
)

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

llm = OpenAI(model="gpt-4o-mini",
             api_key=os.getenv("OPENAI_API_KEY"))
structured_llm_router = llm.with_structured_output(ContextualizeQuestion)


contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

contextualizer = contextualize_q_prompt | structured_llm_router
