import os
import re
import json
import nltk
from timer import Timer
from gensim.models import KeyedVectors, Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
t = Timer()

public_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'public'))

glove_path = os.path.join(public_path, 'glove.6B.300d.txt')

# glove.word2vec.model
# glove.word2vec.model.vectors.npy
glove_word2vec_path = os.path.join(public_path, 'glove.word2vec.format')

glove_trained_path = os.path.join(public_path, 'glove.trained.model')

# print('Glove path: ', glove_path)

# print('Loading GloVe model...')
# t.start()
# # Carrega o modelo GloVe pré-treinado
# glove_model = KeyedVectors.load_word2vec_format(
#     glove_path, binary=False, no_header=True)

# t.stop()
# t.start()

# print('Saving GloVe model as Word2Vec...')

# # Salva o modelo GloVe como Word2Vec
# glove_model.save_word2vec_format(glove_word2vec_path, binary=False)

# t.stop()
# exit()

anime_file_path = os.path.join(public_path, 'animes.json')
# Carrega o arquivo JSON
with open(anime_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Lista de stop words em inglês
stop_words = set(stopwords.words('english'))

# Função de pré-processamento do texto

print('Loading data...')

t.start()


def preprocess_text(text):
    # Converte para minúsculas
    text = text.lower()
    # Remove caracteres especiais e números
    text = re.sub(r'[^a-z ]+', '', text)
    # Tokenização
    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [token for token in tokens if token not in stop_words]
    return tokens


print('Data loaded. Number of texts: ', len(data['content']))

print('Preprocessing texts...')
# Lista com os textos de sinopses pré-processados
processed_texts = [preprocess_text(content) for content in data['content']]

t.stop()

print('Building model...')
t.start()
# Cria o modelo Word2Vec a partir do modelo GloVe pré-treinado
w2v_model = Word2Vec(vector_size=300, workers=4)

w2v_model.build_vocab(processed_texts)

t.stop()
t.start()
print('Training model...')

w2v_model.wv.intersect_word2vec_format(
    glove_word2vec_path, binary=False)
# Treina o modelo Word2Vec com os textos de sinopses
w2v_model.train(processed_texts, total_examples=len(
    processed_texts), epochs=300)


t.stop()

print('Model created. Shape: ', w2v_model.wv.vectors.shape)

# print('Example vector: ', w2v_model.wv[0])
# Salva o modelo Word2Vec treinado
w2v_model.save(glove_trained_path)
