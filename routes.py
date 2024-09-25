from fastapi import APIRouter
from pydantic import BaseModel
from models import Title_with_content_batch
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
import json

from unsupervised.process_data import DataProcessor
dp = DataProcessor() 
@router.post("/batch-title-detection")
async def process_batch_text(input_data: Title_with_content_batch):
    data=input_data.data
    data = [d.model_dump() for d in data]
    processed_data = await dp.process_data(data)
    return {"data": processed_data}