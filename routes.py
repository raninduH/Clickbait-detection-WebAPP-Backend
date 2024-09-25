from fastapi import APIRouter
from models import Titles
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf


router = APIRouter()


# Load the supervised model (either .h5 or .keras)
supervised_model = tf.keras.models.load_model('supervised_clickbait_detection_model.h5')

# POST route to receive an array of titles
@router.post("/single-title-detection")
async def process_text(input_data: Titles):
    # Extract the titles from the request
    received_titles = input_data.titles
    
    vocab_size = 10000  # Maximum number of words in the vocabulary
    max_len = 50  # Maximum number of words in a title (padding/truncating length)

    # Initialize and fit the tokenizer
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(received_titles)

    # convert the titles to sequence
    sequences = tokenizer.texts_to_sequences(received_titles)

    # pad the sequences to the same length
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

    # perform prediction  
    predicted_labels = supervised_model.predict(padded_sequences)
    
    # process -> 0 = "relevant", 1 = "irrelevant"
    # processed_predicted_labels = ["relevant" if pred==0 else "irrelevant" for pred in predicted_labels ]
    # Convert probabilities to binary predictions
    threshold = 0.5
    binary_predictions = (predicted_labels >= threshold).astype(int)
    
    return {"predicted_labels": binary_predictions} 
