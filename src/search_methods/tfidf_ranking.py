import os
from sklearn.feature_extraction.text import TfidfVectorizer

# from timer import Timer
# from preprocess import preprocess_text, read_animes_json
# from ranking import cos_similarity_top_results, euclidean_distance_top_results, calculate_bleu_1_score_for_texts

from src.search_methods.timer import Timer
from src.search_methods.preprocess import preprocess_text, read_animes_json
from src.search_methods.ranking import cos_similarity_top_results, euclidean_distance_top_results, calculate_bleu_1_score_for_texts


def load_tfidf_vectors(names, processed_content):
    print('Loading tfidf data...')
    # Pré-processa as sinopses
    processed_texts = [' '.join(text) for text in processed_content]

    # Calcula os vetores TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    text_vectors = tfidf_matrix.toarray()

    print('Data loaded. TF-IDF matrix shape: ', tfidf_matrix.shape, '\n')
    # Armazena os dados em uma estrutura de dados para ser usada na busca
    return {'titles': names, 'sinopsis': processed_texts, 'tfidf_matrix': tfidf_matrix, 'vectorizer': vectorizer, 'raw_text_vectors': text_vectors}


# Função que realiza a busca
def get_similarity_ranking(query, anime_data, rank_count=10, similarity_method='cosine'):
    t = Timer()
    t.start()
    print('Searching for: "', query, '" using',
          similarity_method, 'similarity')
    # Pré-processa a consulta
    query = ' '.join(preprocess_text(query))
    # Transforma a consulta em vetor TF-IDF
    query_vector = anime_data['vectorizer'].transform([query])

    ranking = []
    if similarity_method == 'cosine':
        ranking = cos_similarity_top_results(
            query_vector, anime_data['tfidf_matrix'], anime_data['titles'], rank_count)

    elif similarity_method == 'euclidean':
        # vetores redimensionados a 1D para serem
        # usados no calculo de distância euclidiana
        query_vector = query_vector.toarray()[0]
        text_vectors = anime_data['raw_text_vectors']

        ranking = euclidean_distance_top_results(
            query_vector, text_vectors, anime_data['titles'], rank_count)

    t.stop()
    return ranking


def get_bleu_ranking(query, anime_data, rank_count=10):
    t = Timer()
    t.start()
    print('Searching for: ', query)
    # Pré-processa a consulta
    query = ' '.join(preprocess_text(query))
    # Transforma a consulta em vetor TF-IDF
    query_vector = anime_data['vectorizer'].transform([query])

    ranking = calculate_bleu_1_score_for_texts(
        anime_data['titles'], anime_data['sinopsis'], query, query_vector, anime_data['raw_text_vectors'], rank_count)

    t.stop()
    return ranking


class TfIdfRanking:
    def __init__(self, names, sinopsis):
        processed_content = [preprocess_text(text) for text in sinopsis]
        self.anime_data = load_tfidf_vectors(names, processed_content)

    def search(self, query, similarity_method, rank_count=10):
        return get_similarity_ranking(query, self.anime_data, rank_count, similarity_method)

    def search_bleu(self, query, rank_count=10):
        return get_bleu_ranking(query, self.anime_data, rank_count)
