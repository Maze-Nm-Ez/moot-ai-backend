from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from RAG.graph import app as rag_app
from RAG.compare_graph import app as compare_app

app = FastAPI()


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
async def test():
    from pprint import pprint

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
    pprint(value["generation"])
    return value["generation"]


@app.post("/compare")
async def compare(request: dict):
    from pprint import pprint

    instructions = request.get("instructions")
    field = request.get("field")

    compare_2019 = request.get("compare_2019")

    if not instructions and not field:
        return {"error": "Either instructions or field must be provided"}
    
    if not instructions and field == 'misc':
        return {"error": "Miscellaneous field requires instructions"}
    
    candidates = []
    if request.get("namal") is True:
        candidates.append("namal rajapakse")
    if request.get("sajith") is True:
        candidates.append("sajith premadasa")
    if request.get("ranil") is True:
        candidates.append("ranil wickramasinghe")
    if request.get("anura") is True:
        candidates.append("anura kumara")

    if len(candidates) == 0:
        return {"error": "At least one candidate must be provided"}
    
    field_instructions = f"What are the key differences between candidates on approaching the {field}?"

    question = f"""You need to focus on the following candidates: {' '.join(candidates)}. 
    You need to answer the question based on the provided instructions and the field.
    {"Also compare with the 2019 election manifesto of the candidates" if compare_2019 else ""}
    {instructions if instructions else field_instructions}"""

    print(question)

    inputs = {
        "question": question
    }
    
    for output in compare_app.stream(inputs):
        for key, value in output.items():
            pprint(f"Node '{key}':")
            pprint(value, indent=2, width=80, depth=None)
        pprint("\n---\n")

    # Final generation
    pprint(value["generation"])
    return {"answer": value["generation"]}

@app.post("/chat")
async def chat(request: dict):
    from fastapi.responses import StreamingResponse
    from pprint import pprint
    import asyncio

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
                    print("PRINTING TRANSLATED GENERATION: ", value["translated_generation"])
                    yield f"{value['translated_generation']}\n\n"  # Stream the generated response
            await asyncio.sleep(0)  # Yield control to the event loop

    return StreamingResponse(event_stream(), media_type="text/event-stream")
