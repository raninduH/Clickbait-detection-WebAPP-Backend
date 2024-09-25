class DataPreProcessor:
    def __init__(self,stmodel):
        self.model = stmodel


    def preprocess(self,data):
        embeddings = []
        for d in data:
            topic_embedding = self.model.encode(d['title'])
            paragraph_embedding = self.model.encode(d['content'])
            embeddings.append((topic_embedding, paragraph_embedding))
        return embeddings