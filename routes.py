from fastapi import APIRouter
from pydantic import BaseModel
from models import Title_with_content_batch ,Titles
import numpy as np
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

router = APIRouter()


# Load the supervised model (either .h5 or .keras)
supervised_model = tf.keras.models.load_model('supervised_clickbait_detection_model_V2.h5')

# POST route to receive an array of titles
@router.post("/single-title-detection")
async def process_text(input_data: Titles):
    # Extract the titles from the request
    received_titles = input_data.titles
    
    max_len = 50  # Maximum number of words in a title (padding/truncating length)

    with open('tokenizer.json', 'r') as json_file:
        tokenizer_json = json_file.read()

    tokenizer = tokenizer_from_json(tokenizer_json)

    # convert the titles to sequence
    sequences = tokenizer.texts_to_sequences(received_titles)

    # pad the sequences to the same length
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

    # perform prediction  
    predicted_labels = supervised_model.predict(padded_sequences)
    
    # process -> 0 = "relevant", 1 = "irrelevant"
    # processed_predicted_labels = ["relevant" if pred==0 else "irrelevant" for pred in predicted_labels ]

    # Convert probabilities to binary predictions
    threshold = 0.6
    binary_predictions = (predicted_labels >= threshold).astype(int)
    binary_predictions = binary_predictions.tolist()
    
    return {"predicted_labels": binary_predictions} 


import json

from unsupervised.process_data import DataProcessor
dp = DataProcessor() 
@router.post("/batch-title-detection")
async def process_batch_text(input_data: Title_with_content_batch):
    data=input_data.data
    data = [d.model_dump() for d in data]
    processed_data = await dp.process_data(data)
    return {"data": processed_data}
