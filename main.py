"""
main.py

This module serves as the entry point for the FastAPI application, integrating various
components of the RAG (Retrieval-Augmented Generation) framework. It sets up the 
FastAPI server, configures CORS middleware, and defines the main endpoint for 
processing user queries through the RAG workflow.

Key functionalities include:
- Loading environment variables from a .env file.
- Handling CORS for cross-origin requests.
- Streaming responses from the RAG application based on user inputs.

Dependencies:
- FastAPI
- dotenv
- RAG components
"""


from pprint import pprint
import logging
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from RAG.graph import app as rag_app
logging.basicConfig(level=logging.CRITICAL)
app = FastAPI()

load_dotenv()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
async def test():
    # Initialize value to ensure it's defined
    value = {}

    # Run
    inputs = {
        "question": "What is the differnce between sajith premadasa's actions for the health sector and ranil wickramasinghe's actions for the health sector?",
        "language": "en"
    }

    config = {"configurable": {"thread_id": "1234"}}
    for output in rag_app.stream(inputs, config=config):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            pprint(value, indent=2, width=80, depth=None)
        pprint("\n---\n")

    # Final generation
    pprint(value.get("generation", "No generation found"))
    return value.get("generation", "No generation found")


@app.post("/chat")
async def chat(request: dict):

    question = request.get("question")
    thread_id = request.get("thread_id")
    language = request.get("language", "en")

    inputs = {
        "question": question,
        "language": language
    }

    config = {"configurable": {"thread_id": thread_id}}

    async def event_stream():
        for output in rag_app.stream(inputs, config=config):
            print("PRINTING OUTPUT: ", output)
            for key, value in output.items():
                pprint(f"Node '{key}':")
                pprint(value, indent=2, width=80, depth=None)
                if "translated_generation" in value:
                    print("PRINTING TRANSLATED GENERATION: ",
                          value["translated_generation"])
                    # Stream the generated response
                    yield f"{value['translated_generation']}\n\n"
            await asyncio.sleep(0)  # Yield control to the event loop

    return StreamingResponse(event_stream(), media_type="text/event-stream")
