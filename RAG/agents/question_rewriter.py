from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_mistralai import ChatMistralAI
import os

# LLM
# llm = ChatOpenAI(model="gpt-4o", temperature=0)
mistral_api_key = os.getenv("MISTRAL_API_KEY")
if not mistral_api_key:
    raise ValueError("MISTRAL_API_KEY environment variable not set")
llm = ChatMistralAI(model="mistral-large-latest", api_key=os.getenv("MISTRAL_API_KEY"))

# Prompt
system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()
