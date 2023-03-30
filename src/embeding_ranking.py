import numpy as np
import pandas as pd
import json
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Carrega os dados
animes_file_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'public', 'animes.json'))

with open(animes_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)


names = data['names']  # array com os títulos dos textos
content = data['content']  # array com os textos

# Define as funções de pré-processamento


def preprocess(text):
    # Tokenização
    tokens = word_tokenize(text.lower())

    # Remoção de stopwords
    stoplist = stopwords.words('english')
    tokens = [token for token in tokens if token not in stoplist]

    # Stemming
    stemmer = SnowballStemmer('english')
    stems = [stemmer.stem(token) for token in tokens]

    # Retorna o texto a partir dos tokens processados
    return stems


# Aplica o pré-processamento nos textos
processed_content = [preprocess(text) for text in content]

print('Data loaded. Number of texts: ', len(processed_content))
# Treina o modelo Word2Vec
vector_size = 200  # Define o tamanho dos vetores de saída do modelo Word2Vec
model = Word2Vec(min_count=1,
                 window=3,
                 vector_size=vector_size,
                 sample=6e-5,
                 alpha=0.03,
                 min_alpha=0.0007,
                 negative=20)

model.build_vocab(processed_content)

print('Training model...')
model.train(processed_content, total_examples=len(
    processed_content), epochs=300)

print('Model created. Shape: ', model.wv.vectors.shape)


def build_text_vectors(processed_content, model):
    # Converte cada texto para um vetor Word2Vec
    print('Building text vectors...')
    print('Text examples: ', processed_content[0], '...')

    vectors = []
    for i, text in enumerate(processed_content):
        text_vector = np.zeros(vector_size)
        for token in text:
            if token in model.wv:
                text_vector += model.wv[token]
        # Dividindo pelo comprimento euclidiano para normalização
        text_vector /= np.linalg.norm(text_vector)
        vectors.append([i, text_vector])

    return vectors


text_vectors = build_text_vectors(processed_content, model)
print('Text vectors created. Shape: ', len(text_vectors))

# Define a função de busca


def search(query, names, text_vectors, model, top_k=10):
    print('Searching for: ', query)
    # Pré-processa a consulta
    processed_query = preprocess(query)

    # Converte a consulta para um vetor Word2Vec
    query_vector = np.zeros(vector_size)
    for token in processed_query:
        if token in model.wv:
            query_vector += model.wv[token]

    # Calcula as similaridades entre a consulta e os textos
    similarity_list = []

    for text_vector in text_vectors:
        index = text_vector[0]
        similarity = cosine_similarity([query_vector], [text_vector[1]])[0][0]
        similarity_list.append([index, similarity])

    print('Similarities calculated. example: ', similarity_list[0])

    similarity_list.sort(key=lambda x: x[1], reverse=True)

    # Retorna os resultados
    results = []
    for i in range(top_k):
        index = similarity_list[i][0]
        results.append([names[index], similarity_list[i][1]])

    return results


# Exemplo de uso
query = "two brothers enter army to become alchemist"
ranking = search(query, names, text_vectors, model)
print(ranking)
