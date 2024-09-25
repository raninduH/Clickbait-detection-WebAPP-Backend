from fastapi import FastAPI
from routes import router  # Import the router from routes.py

# Initialize the FastAPI app
app = FastAPI()

PREFIX = "/clickbait-detection"

# Include the routes from routes.py
app.include_router(router, prefix=PREFIX)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)