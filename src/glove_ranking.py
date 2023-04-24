import os
import re
import nltk
import numpy as np
import pandas as pd
from numba import cuda
from progress.bar import Bar
from nltk.corpus import stopwords
from gensim.models.keyedvectors import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from nltk.tokenize import word_tokenize

from timer import Timer
nltk.download('stopwords')

# GloVe ------------------------------
documents = ['Machine learning is the study of computer algorithms that improve automatically through experience. \
Machine learning algorithms build a mathematical model based on sample data, known as training data. \
The discipline of machine learning employs various approaches to teach computers to accomplish tasks \
where no fully satisfactory algorithm is available.',
             'Machine learning is closely related to computational statistics, which focuses on making predictions using computers. \
The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning.',
             'Machine learning involves computers discovering how they can perform tasks without being explicitly programmed to do so. \
It involves computers learning from data provided so that they carry out certain tasks.',
             'Machine learning approaches are traditionally divided into three broad categories, depending on the nature of the "signal" \
or "feedback" available to the learning system: Supervised, Unsupervised and Reinforcement',
             'Software engineering is the systematic application of engineering approaches to the development of software. \
Software engineering is a computing discipline.',
             'A software engineer creates programs based on logic for the computer to execute. A software engineer has to be more concerned \
about the correctness of the program in all the cases. Meanwhile, a data scientist is comfortable with uncertainty and variability.\
Developing a machine learning application is more iterative and explorative process than software engineering.'
             ]

documents_df = pd.DataFrame(documents, columns=['documents'])

# removing special characters and stop words from the text
stop_words_l = stopwords.words('english')
documents_df['documents_cleaned'] = documents_df.documents.apply(lambda x: " ".join(re.sub(
    r'[^a-zA-Z]', ' ', w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]', ' ', w).lower() not in stop_words_l))

tfidfvectoriser = TfidfVectorizer()
tfidfvectoriser.fit(documents_df.documents_cleaned)
tfidf_vectors = tfidfvectoriser.transform(documents_df.documents_cleaned)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(documents_df.documents_cleaned)
tokenized_documents = tokenizer.texts_to_sequences(
    documents_df.documents_cleaned)
tokenized_paded_documents = pad_sequences(
    tokenized_documents, maxlen=64, padding='post')
vocab_size = len(tokenizer.word_index)+1


def most_similar(doc_id, similarity_matrix, matrix):
    print(f'Document: {documents_df.iloc[doc_id]["documents"]}')
    print('\n')
    print('Similar Documents:')
    if matrix == 'Cosine Similarity':
        similar_ix = np.argsort(similarity_matrix[doc_id])[::-1]
    elif matrix == 'Euclidean Distance':
        similar_ix = np.argsort(similarity_matrix[doc_id])
    for ix in similar_ix:
        if ix == doc_id:
            continue
        print('\n')
        print(f'Document: {documents_df.iloc[ix]["documents"]}')
        print(f'{matrix} : {similarity_matrix[doc_id][ix]}')
# reading Glove word embeddings into a dictionary with "word" as key and values as word vectors


t = Timer()
t.start()

glove_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'public', 'glove.word2vec.6B.300d.model'))

embeddings_index = KeyedVectors.load_word2vec_format(glove_path, binary=False)

t.stop()
print("glove loaded, size: ", embeddings_index)


# with open(glove_path, 'r', encoding='utf-8') as file:
#     file_size = os.path.getsize(glove_path)
#     progress_bar = Bar('Loading GloVe', max=file_size)
#     for line in file:
#         values = line.split()
#         curr_word = values[0]
#         coefs = np.asarray(values[1:], dtype='float32')
#         embeddings_index[curr_word] = coefs
#         progress_bar.next()
#     progress_bar.finish()


# creating embedding matrix, every row is a vector representation from the vocabulary indexed by the tokenizer index.

embedding_matrix = np.zeros((vocab_size, 300))

for word, i in tokenizer.word_index.items():
    if word in embeddings_index:
        embedding_vector = embeddings_index[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
# tf-idf vectors do not keep the original sequence of words, converting them into actual word sequences from the documents

document_embeddings = np.zeros((len(tokenized_paded_documents), 300))
words = tfidfvectoriser.get_feature_names_out()

for i, _ in enumerate(documents_df):
    for j, _ in enumerate(words):
        document_embeddings[i] += embedding_matrix[tokenizer.word_index[words[j]]
                                                   ]*tfidf_vectors[i][j]

document_embeddings = document_embeddings / \
    np.sum(tfidf_vectors, axis=1).reshape(-1, 1)

print("doc embed shape: ", document_embeddings.shape)

pairwise_similarities = cosine_similarity(document_embeddings)
pairwise_differences = euclidean_distances(document_embeddings)

most_similar(0, pairwise_similarities, 'Cosine Similarity')
most_similar(0, pairwise_differences, 'Euclidean Distance')
