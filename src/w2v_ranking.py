import numpy as np
import json
import os
from timer import Timer
from gensim.models import Word2Vec, KeyedVectors
from ranking import cos_similarity_top_results, euclidean_distance_top_results
from preprocess import preprocess_text, read_animes_json


def train_model(vector_size=200):
    # Carrega os dados
    animes_file_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'public', 'animes.json'))

    with open(animes_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    content = data['content']  # array com os textos

    # Aplica o pré-processamento nos textos
    processed_content = [preprocess_text(text) for text in content]

    print('Data loaded. Number of texts: ', len(processed_content))
    # Treina o modelo Word2Vec

    # Cria o modelo Word2Vec
    # configurações do modelo com base em https://radimrehurek.com/gensim/models/word2vec.html
    model = Word2Vec(min_count=1,
                     window=3,
                     vector_size=vector_size,
                     sample=6e-5,
                     alpha=0.03,
                     min_alpha=0.007)

    model.build_vocab(processed_content)

    print('Training model...')
    t = Timer()
    t.start()
    model.train(processed_content, total_examples=len(
        processed_content), epochs=300)

    t.stop()

    w2v_model_file_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'public', 'word2vec.model'))

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
    print('Searching for: "', query, '" using', similarity_method, 'similarity')
    # Pré-processa a consulta
    processed_query = preprocess_text(query)

    # Converte a consulta para um vetor Word2Vec
    query_vector = np.zeros(vector_size)
    for token in processed_query:
        if token in model.wv:
            query_vector = np.add(query_vector, model.wv[token])

    # Calcula as similaridades entre a consulta e os textos

    # Redimensiona o vetor da consulta para que
    # ele possa ser usado na função cosine_similarity
    ranking = []

    if similarity_method == 'cosine':
        query_vector = query_vector.reshape(1, -1)

        ranking = cos_similarity_top_results(
            query_vector, text_vectors, names, top_k)

    elif similarity_method == 'euclidean':
        ranking = euclidean_distance_top_results(
            query_vector, text_vectors, names, top_k)

    t.stop()
    return ranking


class WordToVecRanking:
    def __init__(self, names, processed_content):
        self.names = names  # array com os títulos dos textos

        self.processed_content = processed_content

        self.vector_size = 200  # Define o tamanho dos vetores de saída do modelo Word2Vec

        w2v_model_file_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'public', 'word2vec.model'))

        self.model = KeyedVectors.load(w2v_model_file_path)

        self.text_vectors = build_text_vectors(
            self.processed_content, self.model, self.vector_size)

    def search(self, query, similarity_method, top_k=10):
        return search(query, self.names, self.text_vectors, self.model, self.vector_size, top_k, similarity_method)


# usage example
# anime_data = read_animes_json()
# s_query = 'two brothers enter army to become alchemist'
# ranking = WordToVecRanking(
#     anime_data[0], anime_data[1]).search(s_query, 'euclidean')
# print(ranking)
