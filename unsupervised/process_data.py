from sentence_transformers import SentenceTransformer
from unsupervised.preprocess_data import DataPreProcessor
from unsupervised.cosine_similarities import compute_similarities
import joblib
import numpy as np

class DataProcessor:
    def __init__(self):
        self.kmeans = joblib.load('./unsupervised/kmeans_model.joblib')
        self.stmodel = SentenceTransformer('all-MiniLM-L6-v2')
        self.dpp = DataPreProcessor(self.stmodel)

    async def process_data(self,data):
        data_embeddings = self.dpp.preprocess(data)
        similarities = compute_similarities(data_embeddings)
        clusters = self.kmeans.predict(np.array(similarities).reshape(-1, 1))
        for i, d in enumerate(data):
            d['clickbait_flag'] = True if clusters[i]==1 else False
        return data
    


if __name__ == "__main__":
    data=[
    {
        "title": "The rise of renewable energy",
        "content": "Solar and wind power are becoming more cost-effective and are being adopted globally to combat climate change."
    },
    {
        "title": "The benefits of physical exercise",
        "content": "Regular exercise can help reduce the risk of chronic diseases, improve mental health, and boost longevity."
    },
    {
        "title": "The future of space exploration",
        "content": "Space agencies like NASA are planning missions to Mars and beyond, with the aim of establishing human colonies on other planets."
    },
    {
        "title": "The rise of renewable energy",
        "content": "Artificial Intelligence is being increasingly used in industries like healthcare and finance to automate decision-making processes."
    },
    {
        "title": "The future of space exploration",
        "content": "Global warming is accelerating the melting of polar ice caps, leading to rising sea levels and threatening coastal communities."
    },
    {
        "title": "Artificial Intelligence in education",
        "content": "AI-powered tools are helping teachers create personalized learning plans for students, enhancing the overall learning experience."
    },
    {
        "title": "The impact of social media on mental health",
        "content": "Prolonged exposure to social media platforms like Instagram and Facebook has been linked to anxiety, depression, and other mental health issues."
    },
    {
        "title": "The benefits of physical exercise",
        "content": "Quantum computing is set to revolutionize industries like cryptography and material science by solving complex problems much faster than classical computers."
    },
    {
        "title": "Artificial Intelligence in education",
        "content": "Renewable energy sources like wind and solar power are crucial to reducing global carbon emissions and combatting climate change."
    }
    ]    
    dp = DataProcessor()
    import asyncio

    async def main():
        dp = DataProcessor()
        d = await dp.process_data(data)
        print(d)

    asyncio.run(main())