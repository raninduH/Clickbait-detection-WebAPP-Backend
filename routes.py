from fastapi import APIRouter
from models import Title_with_content_batch ,Titles
from supervised.title_prediction import predcit_titles


router = APIRouter()


# POST route to receive an array of titles
@router.post("/titles-only-detection")
async def process_title(input_data: Titles):
    # Extract the titles from the request
    received_titles = input_data.titles
    predicted_labels = predcit_titles(received_titles)
    
    return {"predicted_labels": predicted_labels} 


from unsupervised.process_data import DataProcessor
dp = DataProcessor() 
@router.post("/batch-title-detection")
async def process_batch_text(input_data: Title_with_content_batch):
    data=input_data.data
    data = [d.model_dump() for d in data]
    processed_data = await dp.process_data(data)
    return {"data": processed_data}
