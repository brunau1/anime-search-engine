import os
import json
import nltk
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
    return stems


def read_animes_json():
    animes_file_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'public', 'animes.json'))

    with open(animes_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    names = data['names']  # array com os títulos dos textos
    content = data['content']  # array com os textos

    processed_content = [
        preprocess_text(text) for text in content]

    return names, processed_content
