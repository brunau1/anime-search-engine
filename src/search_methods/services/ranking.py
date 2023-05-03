import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance

from services.preprocess import simple_preprocess_text, preprocess_text


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


def calculate_bleu_1_score_for_texts(titles, texts, search_phrase, search_encoded, texts_embeddings, top_k=10):
    similarity_matrix = np.inner(search_encoded, texts_embeddings)

    similar_text_indices = np.argsort(similarity_matrix)[::-1]

    curr_scores = []
    for i in range(0, top_k):
        text_idx = similar_text_indices[i]

        reference = preprocess_text(texts[text_idx])
        candidate = preprocess_text(search_phrase)

        # remove from the reference the words that are not in the candidate
        reference = [word for word in reference if word in candidate]

        reference = list(dict.fromkeys(reference))

        score = sentence_bleu([reference], candidate, weights=(1, 0, 0, 0))

        print(
            f"BLEU score for '{' '.join(reference)}' and '{' '.join(candidate)}': {score}")

        curr_scores.append(
            {'idx': text_idx, 'title': titles[text_idx],
             'cos_sim': np.float64(similarity_matrix[text_idx]), 'bleu_score': score})

    return curr_scores
