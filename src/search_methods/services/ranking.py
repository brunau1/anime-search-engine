import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance


def cos_similarity_top_results(query_vector, text_vectors, names, top_k=10):
    cosine_similarities = cosine_similarity(
        query_vector, text_vectors).flatten()

    most_similar_indexes = cosine_similarities.argsort()[
        ::-1][:int(top_k)]

    print(most_similar_indexes)

    ranking = []

    for index in most_similar_indexes:
        anime_title = names[index]
        similarity_score = cosine_similarities[index]
        ranking.append([anime_title, similarity_score])

    return ranking

# calcula a distancia euclidiana entre o vetor de busca
# e os vetores de cada anime


def euclidean_distance_top_results(query_vector, text_vectors, names, top_k=10):
    distances = []
    for text_vector in text_vectors:
        d_value = distance.euclidean(query_vector, text_vector)
        distances.append(d_value)

    most_similar_indexes = np.argsort(distances)[:int(top_k)]

    ranking = []

    for index in most_similar_indexes:
        anime_title = names[index]
        similarity_score = distances[index]
        ranking.append([anime_title, similarity_score])

    return ranking
