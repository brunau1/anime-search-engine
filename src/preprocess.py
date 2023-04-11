import os
import re
import json
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
nltk.download('stopwords')
nltk.download('punkt')


def preprocess_text(text=''):

    # Remove pontuação
    text = text.translate(str.maketrans('', '', string.punctuation))
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
    print('Reading animes.json...')
    animes_file_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'public', 'animes.json'))

    with open(animes_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    names = data['names']  # array com os títulos dos textos
    content = data['content']  # array com os textos

    print('preprocessing text...')

    processed_content = [
        word_tokenize(re.sub(r'[.,"\'-?:!;]', '', names[i].lower())) + preprocess_text(text.lower()) for i, text in enumerate(content)
    ]

    return names, processed_content
