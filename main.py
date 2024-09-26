from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from routes import router  # Import the router from routes.py

# Initialize the FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend's origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)
PREFIX = "/clickbait-detection"

# Include the routes from routes.py
app.include_router(router, prefix=PREFIX)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)