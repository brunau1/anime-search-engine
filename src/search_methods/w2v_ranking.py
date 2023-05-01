import os
import json
import numpy as np
from gensim.models import Word2Vec, KeyedVectors

# from services.preprocess import simple_preprocess_text
# from services.ranking import cos_similarity_top_results, euclidean_distance_top_results
# from services.timer import Timer

from src.search_methods.services.timer import Timer
from src.search_methods.services.ranking import cos_similarity_top_results, euclidean_distance_top_results
from src.search_methods.services.preprocess import simple_preprocess_text

VECTOR_SIZE = 200


def train_model(vector_size=200):
    # Carrega os dados
    animes_file_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'public', 'dataset', 'animes.json'))

    with open(animes_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    content = data['content']  # array com os textos

    # Aplica o pré-processamento nos textos
    processed_content = [simple_preprocess_text(text) for text in content]

    print('Data loaded. Number of texts: ', len(processed_content))
    # Treina o modelo Word2Vec

    # Cria o modelo Word2Vec
    # configurações do modelo com base em https://radimrehurek.com/gensim/models/word2vec.html
    model = Word2Vec(vector_size=vector_size,
                     sample=6e-5,
                     alpha=0.03,
                     min_alpha=0.007,
                     workers=4)

    model.build_vocab(processed_content)

    print('Training model...')
    t = Timer()
    t.start()

    model.train(processed_content, total_examples=len(
        processed_content), epochs=300)

    t.stop()

    w2v_model_file_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'public', 'models', 'word2vec.model'))

    model.save(w2v_model_file_path)

    print('Model created. Shape: ', model.wv.vectors.shape)


def build_text_vectors(processed_content, model, vector_size):
    # Converte cada texto para um vetor Word2Vec
    print('Building text vectors...')
    vectors = []
    for text in processed_content:
        text_vector = np.zeros(vector_size)
        for token in text:
            if token in model.wv:
                text_vector = np.add(text_vector, model.wv[token])
        vectors.append(text_vector)

    print('Text vectors created. Shape: ', np.array(vectors).shape)
    return vectors

# Define a função de busca


def search(query, names, text_vectors, model, vector_size, top_k=10, similarity_method='cosine'):
    t = Timer()
    t.start()
    print('Searching for: "', query, '" using',
          similarity_method, 'similarity')
    # Pré-processa a consulta
    processed_query = simple_preprocess_text(query)

    # Converte a consulta para um vetor Word2Vec
    query_vector = np.zeros(vector_size)
    for token in processed_query:
        if token in model.wv:
            query_vector = np.add(query_vector, model.wv[token])

    # Calcula as similaridades entre a consulta e os textos

    ranking = []

    if similarity_method == 'cosine':
        # Redimensiona o vetor da consulta para que
        # ele possa ser usado na função cosine_similarity
        query_vector = query_vector.reshape(1, -1)

        ranking = cos_similarity_top_results(
            query_vector, text_vectors, names, top_k)

    elif similarity_method == 'euclidean':
        ranking = euclidean_distance_top_results(
            query_vector, text_vectors, names, top_k)

    t.stop()
    return ranking


class WordToVecRanking:
    def __init__(self, names, sinopsis):
        self.names = names  # array com os títulos dos textos

        self.processed_content = [
            simple_preprocess_text(text) for text in sinopsis]

        # Define o tamanho dos vetores de saída do modelo Word2Vec
        self.vector_size = VECTOR_SIZE

        w2v_model_file_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', 'public', 'models', 'word2vec.model'))

        print('Loading model from: ', w2v_model_file_path)
        self.model = KeyedVectors.load(w2v_model_file_path)

        self.text_vectors = build_text_vectors(
            self.processed_content, self.model, self.vector_size)

    def search(self, query, similarity_method, top_k=10):
        return search(query, self.names, self.text_vectors, self.model, self.vector_size, top_k, similarity_method)


# usage example
# anime_data = read_animes_json()
# s_query = 'two brothers enter army to become alchemist'
# ranking = WordToVecRanking(
#     anime_data[0], anime_data[1])
# print(ranking)

# train_model()
