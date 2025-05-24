import os
from dotenv import load_dotenv
import uvicorn

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    # Run the FastAPI app using uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
