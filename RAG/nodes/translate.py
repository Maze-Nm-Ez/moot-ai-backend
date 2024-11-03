from typing import Annotated, TypedDict
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
import os
from langchain_mistralai import ChatMistralAI
class TranslationInput(TypedDict):
    generation: str
    language: str

class TranslationOutput(TypedDict):
    translated_generation: str

mistral_api_key = os.getenv("MISTRAL_API_KEY")
if not mistral_api_key:
    raise ValueError("MISTRAL_API_KEY environment variable not set")
llm = ChatMistralAI(model="mistral-large-latest", api_key=os.getenv("MISTRAL_API_KEY"))

def translate(state: Annotated[TranslationInput, "TranslationInput"]) -> TranslationOutput:
    #llm = ChatOpenAI(model="gpt-4o")    
    generation = state["generation"]
    target_language = state["language"]

    if target_language == "si":
        target_language = "Sinhala"
    elif target_language == "ta":
        target_language = "Tamil"
    elif target_language == "en":
        target_language = "English"
    elif target_language == "seng":
        target_language = "Singlish"
    elif target_language == "slang":
        target_language = "Gen Z Slang"
    
    prompt = f"Translate the following text to {target_language}. If it's already in {target_language}, just return it as is. Do not say it's already in {target_language} Return the same thing please:\n\n{generation}"
    
    response = llm.invoke(prompt)
    
    return {"translated_generation": response.content}