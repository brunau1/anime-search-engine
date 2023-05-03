import os
import re
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
nltk.download('stopwords')
nltk.download('punkt')


def bert_simple_preprocess_text(text=''):
    text = text.lower()
    # Remove caracteres especiais e números
    text = re.sub(r'[^a-z ]+', '', text)

    return text


def simple_preprocess_text(text=''):
    stop_words = set(stopwords.words('english'))
    # Converte para minúsculas
    text = text.lower()
    # Remove caracteres especiais e números
    text = re.sub(r'[^a-z ]+', '', text)
    # Tokenização
    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [token for token in tokens if token not in stop_words]
    return tokens


def preprocess_text(text=''):
    stop_words = set(stopwords.words('english'))
    # Converte para minúsculas
    text = text.lower()

    # Remove pontuação
    text = re.sub(r'[^a-z ]+', '', text)
    # Tokenização
    tokens = word_tokenize(text)

    # Remoção de stopwords
    tokens = [token for token in tokens if token not in stop_words]

    # Stemming
    stemmer = SnowballStemmer('english')
    stems = [stemmer.stem(token) for token in tokens]

    # Reconstroi o texto a partir dos tokens processados
    return stems


def read_animes_json(path=''):
    print('Reading animes.json...')
    animes_file_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'public', 'dataset', 'animes.json')) if path == '' else path

    with open(animes_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    names = data['names']  # array com os títulos dos textos
    content = data['content']  # array com os textos

    print('adding titles to memory...')

    for i, text in enumerate(content):
        curr_title = names[i]
        content[i] = text + ' ' + curr_title

    return names, content
