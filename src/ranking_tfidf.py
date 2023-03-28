import os
import nltk
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
nltk.download('punkt')


def preprocess_text(text):
    # Tokenização
    tokens = word_tokenize(text.lower())

    # Remoção de stopwords
    stoplist = stopwords.words('english')
    tokens = [token for token in tokens if token not in stoplist]

    # Stemming
    stemmer = SnowballStemmer('english')
    stems = [stemmer.stem(token) for token in tokens]

    # Reconstroi o texto a partir dos tokens processados
    return ' '.join(stems)


# Carrega os dados do arquivo JSON e faz o pré-processamento
def load_tfidf_vectors():
    print('Loading data...')
    file_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'public', 'animes.json'))

    with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # Separa os títulos e sinopses em listas separadas
    titles = json_data['names']
    contents = json_data['content']

    # Pré-processa as sinopses
    processed_texts = [preprocess_text(text) for text in contents]

    # Calcula os vetores TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_texts)

    print('Data loaded. TF-IDF matrix shape: ', tfidf_matrix.shape, '\n')
    # Armazena os dados em uma estrutura de dados para ser usada na busca
    return {'titles': titles, 'tfidf_matrix': tfidf_matrix, 'vectorizer': vectorizer}


# Função que realiza a busca
def get_ranking_by_cos_similarity(query, rank_count, anime_data):
    # Pré-processa a consulta
    query = preprocess_text(query)
    # Transforma a consulta em vetor TF-IDF
    query_vector = anime_data['vectorizer'].transform([query])
    # Calcula a similaridade coseno entre a consulta e as sinopses
    cosine_similarities = cosine_similarity(
        query_vector, anime_data['tfidf_matrix']).flatten()
    # Obtém os índices dos animes mais similares
    most_similar_indexes = cosine_similarities.argsort()[
        ::-1][:int(rank_count)]
    # Cria um ranking com os títulos dos animes mais similares
    ranking = []
    for index in most_similar_indexes:
        anime_title = anime_data['titles'][index]
        similarity_score = cosine_similarities[index]
        ranking.append((anime_title, similarity_score))
    return ranking
