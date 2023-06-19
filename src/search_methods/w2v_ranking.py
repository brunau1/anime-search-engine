import os
import json
import numpy as np
from gensim.models import Word2Vec, KeyedVectors

# from preprocess import preprocess_text, read_animes_json, preprocess_text
# from ranking import cos_similarity_top_results, euclidean_distance_top_results, calculate_bleu_1_score_for_texts
# from timer import Timer

from src.search_methods.timer import Timer
from src.search_methods.ranking import cos_similarity_top_results, euclidean_distance_top_results, calculate_bleu_1_score_for_texts
from src.search_methods.preprocess import preprocess_text

VECTOR_SIZE = 200


def train_model(vector_size=200):
    # Carrega os dados
    animes_file_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'public', 'dataset', 'animes.json'))

    with open(animes_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    content = data['content']  # array com os textos

    # Aplica o pré-processamento nos textos
    processed_content = [preprocess_text(
        text) for text in content if len(text.split()) > 40]

    print('Data loaded. Number of texts: ', len(processed_content))
    model = Word2Vec(min_count=3,  # 3
                     window=5,  # 5
                     sample=1e-5,  # 1e-5
                     vector_size=vector_size,
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
        os.path.dirname(__file__), '..', '..', 'public', 'models', 'word2vec.model'))

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
    processed_query = preprocess_text(query)

    query_vector = np.zeros(vector_size)
    for token in processed_query:
        if token in model.wv:
            query_vector = np.add(query_vector, model.wv[token])

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


def search_and_bleu(query, names, texts, text_vectors, model, vector_size, top_k=10):
    t = Timer()
    t.start()
    print('Searching for: "', query, '"')
    processed_query = preprocess_text(query)

    query_vector = np.zeros(vector_size)
    for token in processed_query:
        if token in model.wv:
            query_vector = np.add(query_vector, model.wv[token])

    query_vector = query_vector.reshape(1, -1)

    ranking = calculate_bleu_1_score_for_texts(
        names, texts, query, query_vector, text_vectors, top_k)

    t.stop()
    return ranking


class WordToVecRanking:
    def __init__(self, names, sinopsis):
        self.names = names  # array com os títulos dos textos
        self.sinopsis = sinopsis  # array com os textos

        self.processed_content = [
            preprocess_text(text) for text in sinopsis]

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

    def search_and_bleu(self, query, top_k=10):
        return search_and_bleu(query, self.names, self.sinopsis, self.text_vectors, self.model, self.vector_size, top_k)
