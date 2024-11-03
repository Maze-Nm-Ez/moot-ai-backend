from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
import os

class OutputFormatter(BaseModel):
    """format the output to be shown in the frontend."""

    namal_text: str = Field(
        ...,
        description="The text to be shown in the frontend for namal.",
    )
    ranil_text: str = Field(
        ...,
        description="The text to be shown in the frontend for ranil.",
    )
    sajith_text: str = Field(
        ...,
        description="The text to be shown in the frontend for sajith.",
    )
    anura_text: str = Field(
        ...,
        description="The text to be shown in the frontend for anura.",
    )
    misc_text: str = Field(
        ...,
        description="The text to be shown in the frontend for the miscellaneous information. like the conclusion",
    )


template = """
You are a very vigilant and helpful journalist. Use the following pieces of 
context to answer the question at the end. Your focus is 2024 presidential election in Sri Lanka.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Be concise and helpful. Your main Task is to do a comparison between the candidates. And between their
past elections if requested. Do a comprehensive comparison. Do not hallucinate. Do not be biased. Do not
make up informations.

________________________________________________________________________________
Here are the web results regarding the question:
{web_context}

Here are the results from the manifesto of the candidates:
________________________________________________________________________________
namel rajapakse:
{namal_context}

________________________________________________________________________________
ranil wickramasinghe:
{ranil_context}

________________________________________________________________________________
sajith premadasa:
{sajith_context}

________________________________________________________________________________
anura kumara:
{anura_context}

________________________________________________________________________________
Question: {question}

Format the output in the following way:
Any information directly related to namal should be in the namal_text.
Any information directly related to ranil should be in the ranil_text.
Any information directly related to sajith should be in the sajith_text.
Any information that is not related to any of the candidates should be in the misc_text. For an example, The conclusion

"""
custom_rag_prompt = PromptTemplate.from_template(template)

# LLM
# llm = ChatOpenAI(model="gpt-4o")
mistral_api_key = os.getenv("MISTRAL_API_KEY")
if not mistral_api_key:
    raise ValueError("MISTRAL_API_KEY environment variable not set")
llm = ChatMistralAI(model="mistral-large-latest", api_key=os.getenv("MISTRAL_API_KEY"))
structured_llm_output = llm.with_structured_output(OutputFormatter)


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = custom_rag_prompt | structured_llm_output