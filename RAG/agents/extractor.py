### Router

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


# Data model
class ExtractQuery(BaseModel):
    """Route a user query to the relevant datasources with subquestions."""

    namal_vector_search_query: str = Field(
        "",
        description="The query to search the vector store of namal.",
    )
    ranil_vector_search_query: str = Field(
        "",
        description="The query to search the vector store of ranil.",
    )
    sajith_vector_search_query: str = Field(
        "",
        description="The query to search the vector store of sajith.",
    )
    anura_vector_search_query: str = Field(
        "",
        description="The query to search the vector store of anura.",
    )
    web_search_query: str = Field(
        "",
        description="The query to search the web.",
    )

llm = ChatOpenAI(model="gpt-4o")
structured_llm_router = llm.with_structured_output(ExtractQuery)

# Prompt
system = """You are an expert at routing a user question to a vectorstore or web search.
There are three vectorstores. One contains documents related to Manifests of political candidate Sajith Premadasa.
Another contains documents related to Manifests of political candidate Namal Rajapaksa.
The third contains documents related to Manifests of political candidate Ranil Wickramasinghe. There is another
related to Anura Kumara Dissanayake.

for an example, what this candidate do for education sector, health sector etc is on the vectorstore.
And also their plans for the future of the country is on the vectorstore.

If the question involves something about a candidate's policies in a past year, then you will have to do a websearch.
And also if you feel like a web search will be usefull. Do a web search.

If it seems unneccesary to search the vectorstores, then don't search the vectorstores. Keep those queries empty.
Keep web search query empty if it is not required.

After deciding,
Output the 'namal_vector_search_query': The query that needs to be searched from the vector store of namal.
And the 'ranil_vector_search_query': The query that needs to be searched from the vector store of ranil.
And the 'sajith_vector_search_query': The query that needs to be searched from the vector store of sajith.
And the 'anura_vector_search_query': The query that needs to be searched from the vector store of anura.
And the 'web_search_query': The query that needs to be searched from the web.
"""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_extractor = route_prompt | structured_llm_router