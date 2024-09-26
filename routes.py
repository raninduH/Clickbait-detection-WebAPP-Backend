from fastapi import APIRouter, HTTPException, UploadFile, File
import json  # Ensure you are using the standard library's json module
from models import Title_with_content_batch ,Titles
from supervised.title_prediction import predict_titles


router = APIRouter()


# POST route to receive an array of titles
@router.post("/titles-only-detection")
async def process_title(input_data: Titles):
    # Extract the titles from the request
    received_titles = input_data.titles
    prediction_results = predict_titles(received_titles)
    
    return {"data": prediction_results}


from unsupervised.process_data import DataProcessor
dp = DataProcessor()

@router.post("/batch-title-detection")
async def process_batch_text(file: UploadFile = File(...)):
    try:
        # Read the file
        file_content = await file.read()

        # Parse the JSON data
        data = json.loads(file_content.decode("utf-8"))

        # Ensure the structure is valid
        processed_data = []
        for item in data:
            # Extract topic and paragraphs
            topic = item['original_topic']
            paragraphs = item['original_paragraphs']
            processed_data.append({
                'title': topic,
                'content': paragraphs
            })

        # Pass the processed data to your processing logic (e.g., dp.process_data)
        # Assuming dp.process_data is an async function that processes the data
        processed_result = await dp.process_data(processed_data)

        # Return the processed data
        return {"data": processed_result}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")
