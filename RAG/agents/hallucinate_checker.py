from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
# from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
import os

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


# LLM with function call
#llm = ChatOpenAI(model="gpt-4o")
mistral_api_key = os.getenv("MISTRAL_API_KEY")
if not mistral_api_key:
    raise ValueError("MISTRAL_API_KEY environment variable not set")
llm = ChatMistralAI(model="mistral-large-latest", api_key=os.getenv("MISTRAL_API_KEY"))
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# Prompt
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader