import numpy as np
import json
import os
import re
from tqdm import tqdm
from gensim.models import Word2Vec, KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

from timer import Timer
from ranking import cos_similarity_top_results, euclidean_distance_top_results
from preprocess import preprocess_text, read_animes_json, simple_preprocess_text

# from src.timer import Timer
# from src.ranking import cos_similarity_top_results, euclidean_distance_top_results
# from src.preprocess import preprocess_text, read_animes_json, simple_preprocess_text

VECTOR_SIZE = 200


def train_model(vector_size=200):
    # Carrega os dados
    animes_file_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'public', 'animes.json'))

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
            os.path.dirname(__file__), '..', 'public', 'word2vec.model'))

        print('Loading model from: ', w2v_model_file_path)
        self.model = KeyedVectors.load(w2v_model_file_path)

        self.text_vectors = build_text_vectors(
            self.processed_content, self.model, self.vector_size)

    def search(self, query, similarity_method, top_k=10):
        return search(query, self.names, self.text_vectors, self.model, self.vector_size, top_k, similarity_method)


# usage example
anime_data = read_animes_json()
s_query = 'two brothers enter army to become alchemist'
# ranking = WordToVecRanking(
#     anime_data[0], anime_data[1])
# print(ranking)
text_vectors = WordToVecRanking(
    anime_data[0][:1000], anime_data[1][:1000]).text_vectors

higger_cosine_similarities = []
mean_cosine_similarities = []
# # Calcula a similaridade entre o texto atual e todos os demais
for i, _ in tqdm(enumerate(text_vectors)):
    curr_text = re.sub(r'[^a-z ]+', '', anime_data[1][i].lower())

    if len(curr_text.split()) > 96:
        sim = cosine_similarity([text_vectors[i]], text_vectors).flatten()
        most_similar_idx = sim.argsort()[
            ::-1][1]
        mean_similar_idx = sim.argsort()[::-1][len(sim)//4]

        if sim[most_similar_idx] < 0.92:
            higger_cosine_similarities.append(
                [i, most_similar_idx, round(sim[most_similar_idx], 4)])

        mean_cosine_similarities.append(
            [i, mean_similar_idx, round(sim[mean_similar_idx], 4)])

        # print('Text: ', i, 'Most similar: ', most_similar_idx,
        #       'Similarity: ', sim[most_similar_idx])

higger_sim_values = np.argsort([score[2]
                               for score in higger_cosine_similarities])[::-1]

higger_mean_sim_values = np.argsort([score[2]
                                    for score in mean_cosine_similarities])[::-1]

# print('Higger cosine similarities: ', higger_cosine_similarities[:5])
# print('Mean cosine similarities: ', mean_cosine_similarities[:5])
# print('Higger sim values len: ', len(higger_sim_values))
# print('similar texts: ', anime_data[1][708], "\nthe second -----: ", anime_data[1][719])

final_similarities = []

for i, _ in enumerate(higger_cosine_similarities):
    if i % 2 == 0:
        final_similarities.append(
            higger_cosine_similarities[higger_sim_values[i]])
    elif i % 3 == 0:
        final_similarities.append(
            mean_cosine_similarities[higger_mean_sim_values[i]])
    else:
        final_similarities.append(
            mean_cosine_similarities[i])

print('Final similarities: ', len(final_similarities))
print('Final similarities ex: ', final_similarities[:10])

public_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'public'))

np.save(os.path.join(public_path, 'dataset',
        'similarities.npy'), final_similarities)

loaded = np.load(os.path.join(public_path, 'dataset', 'similarities.npy'))

print('Similarities saved shape: ', loaded.shape)
# para cada texto ate a metade do array de textos
# calcula a similaridade entre o texto atual e todos os demais não só ate a metade do array
# seleciona o indice do texto com maior similaridade com o texto atual ou
# seleciona o indice do texto com menor similaridade com o texto atual ou
# seleciona o indice do texto com similaridade media com o texto atual
# salva os indices dos textos e a similaridade calculada em um array


# half_vectors1 = text_vectors[:len(text_vectors)//2]
# half_vectors2 = text_vectors[len(text_vectors)//2:]

# cosine_similarities = []

# for i, _ in enumerate(half_vectors1):
#     cosine_similarities.append(cosine_similarity(
#         [half_vectors1[i]], [half_vectors2[i]]).flatten()[0])

# print(len(cosine_similarities))

# results = np.argsort(cosine_similarities)[:10]

# print(results)

# train_model()
