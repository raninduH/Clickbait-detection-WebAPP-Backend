from fastapi import APIRouter
from pydantic import BaseModel

# Define the router
router = APIRouter()

# Define the data model (request body)
class TextInput(BaseModel):
    text: str

# Define the POST route to receive text
@router.post("/single-title-detection")
async def process_text(input_data: TextInput):
    # Extract the text from the request
    received_text = input_data.text
    
    # You can add your processing logic here
    processed_text = received_text.upper()  # Example processing: converting to uppercase

    # Return the response
    return {"received_text": received_text, "processed_text": processed_text}
