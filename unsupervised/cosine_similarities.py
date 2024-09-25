from sklearn.metrics.pairwise import cosine_similarity

def compute_similarities(embeddings):
    similarities = []
    for topic_emb, para_emb in embeddings:
        sim = cosine_similarity([topic_emb], [para_emb])[0][0]
        similarities.append(sim)
    return similarities