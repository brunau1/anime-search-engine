import os
from scipy.spatial.distance import euclidean
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
import torch
from preprocess import read_animes_json
import re
from timer import Timer
from nltk.corpus import stopwords
from tqdm import tqdm

public_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'public'))

data = read_animes_json()

titles = data[0]
documents = data[1][:100]

stop_words_l = stopwords.words('english')
anime_synopses = []

for text in documents:
    text = re.sub(r'[^a-z ]+', '', text.lower())
    anime_synopses.append(text)


# Load the pre-trained BERT model
bert_path = os.path.join(public_path, 'bert_pretrained_model')

model = BertModel.from_pretrained(bert_path).cuda()
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased', do_lower_case=True)

search_query = "two brothers enter army and become alchemist"

search_query = re.sub(r'[^a-z ]+', '', search_query.lower())

# Tokenize all synopses using the BERT tokenizer
input_ids = []
encoded_synopses = []
for synopsis in anime_synopses:
    t_synopsis = tokenizer.encode(synopsis, add_special_tokens=True,
                                  max_length=80,
                                  padding='max_length',
                                  truncation=True,
                                  return_tensors='pt').to('cuda')
    input_ids.append(t_synopsis)

# Stack all tensors into a single tensor
encoded_synopses = torch.cat(input_ids, dim=0)

# Compute the representations of all synopses using the BERT model
with torch.no_grad():
    all_synopsis_representations = model(encoded_synopses)[1].cpu()

print(all_synopsis_representations.shape)

# Compute the representation of the search query using the BERT model
encoded_search_query = tokenizer.encode(search_query, add_special_tokens=True,
                                        padding='max_length', max_length=80, truncation=True, return_tensors='pt').to('cuda')
with torch.no_grad():
    search_query_representation = model(
        encoded_search_query)[1].cpu()

# Compute the cosine similarity between the search query representation and all synopsis representations
cosine_similarities = F.cosine_similarity(
    all_synopsis_representations, search_query_representation, dim=1).numpy().tolist()

# Compute the euclidean distance between the search query representation and all synopsis representations
euclidean_distances = [euclidean(search_query_representation.numpy().
                                 flatten(), x.numpy().flatten()) for x in all_synopsis_representations]

# Get the indices of the top 10 results for both similarity and euclidean distance
top_cosine_similarities_indices = sorted(range(len(
    cosine_similarities)), key=lambda i: cosine_similarities[i], reverse=True)[:10]
top_euclidean_distances_indices = sorted(
    range(len(euclidean_distances)), key=lambda i: euclidean_distances[i])[:10]

# Print the top 10 results for cosine similarity
print("Top 10 results for cosine similarity:")
for i, index in enumerate(top_cosine_similarities_indices):
    print(
        f"{i+1}. title: {titles[index]}, Cosine similarity: {cosine_similarities[index]}")

# Print the top 10 results for euclidean distance
print("\nTop 10 results for euclidean distance:")
for i, index in enumerate(top_euclidean_distances_indices):
    print(
        f"{i+1}. title: {titles[index]}, Euclidean distance: {euclidean_distances[index]}")


# anime_synopses_encoded = []
# for synopsis in anime_synopses:
#     encoded = tokenizer.encode(
#         synopsis, add_special_tokens=True, return_tensors='pt')
#     anime_synopses_encoded.append(encoded)

# # 1 represents the output of the last hidden layer (CLS token)
# search_query_representation = model(search_query_encoded)[1].detach().numpy()
# anime_synopses_representations = []
# for synopsis in anime_synopses_encoded:
#     representation = model(synopsis)[1].detach().numpy()
#     anime_synopses_representations.append(representation)


# similarities = []
# for representation in anime_synopses_representations:
#     similarity = cosine_similarity(search_query_representation, representation)
#     similarities.append(similarity[0][0])

# sorted_anime_synopses = [synopsis for _, synopsis in sorted(
#     zip(similarities, anime_synopses), reverse=True)]
# for synopsis in sorted_anime_synopses:
#     print(synopsis)

# import os
# import re
# import nltk
# import torch
# import numpy as np
# import pandas as pd
# from progress.bar import Bar
# from nltk.corpus import stopwords
# from keras.utils import pad_sequences
# from nltk.tokenize import word_tokenize
# from keras.preprocessing.text import Tokenizer
# from gensim.models.keyedvectors import KeyedVectors
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.metrics.pairwise import euclidean_distances
# from sklearn.feature_extraction.text import TfidfVectorizer

# from preprocess import read_animes_json
# from timer import Timer

# data = read_animes_json()

# titles = data[0]
# documents = data[0]

# # Sample corpus
# # documents = ['Machine learning is the study of computer algorithms that improve automatically through experience.\
# # Machine learning algorithms build a mathematical model based on sample data, known as training data.\
# # The discipline of machine learning employs various approaches to teach computers to accomplish tasks \
# # where no fully satisfactory algorithm is available.',
# #              'Machine learning is closely related to computational statistics, which focuses on making predictions using computers.\
# # The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning.',
# #              'Machine learning involves computers discovering how they can perform tasks without being explicitly programmed to do so. \
# # It involves computers learning from data provided so that they carry out certain tasks.',
# #              'Machine learning approaches are traditionally divided into three broad categories, depending on the nature of the "signal"\
# # or "feedback" available to the learning system: Supervised, Unsupervised and Reinforcement',
# #              'Software engineering is the systematic application of engineering approaches to the development of software.\
# # Software engineering is a computing discipline.',
# #              'A software engineer creates programs based on logic for the computer to execute. A software engineer has to be more concerned\
# # about the correctness of the program in all the cases. Meanwhile, a data scientist is comfortable with uncertainty and variability.\
# # Developing a machine learning application is more iterative and explorative process than software engineering.'
# #              ]

# documents_df = pd.DataFrame(documents, columns=['documents'])

# # removing special characters and stop words from the text
# stop_words_l = stopwords.words('english')
# documents_df['documents_cleaned'] = documents_df.documents.apply(lambda x: " ".join(re.sub(
#     r'[^a-zA-Z]', ' ', w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]', ' ', w).lower() not in stop_words_l))

# # tfidfvectoriser = TfidfVectorizer()
# # tfidfvectoriser.fit(documents_df.documents_cleaned)
# # tfidf_vectors = tfidfvectoriser.transform(documents_df.documents_cleaned)

# # pairwise_similarities = np.dot(tfidf_vectors, tfidf_vectors.T)
# # pairwise_differences = euclidean_distances(tfidf_vectors)


# def most_similar(query, similarity_matrix, matrix):
#     print(f'Query: {query}')
#     print('\n')
#     print('Similar Documents:')
#     if matrix == 'Cosine Similarity':
#         sim_list = np.argsort(similarity_matrix)[::-1][:5]
#     elif matrix == 'Euclidean Distance':
#         sim_list = np.argsort(similarity_matrix)[:5]
#     for pos in sim_list:
#         print('\n')
#         print(f'Document: {titles[pos]}')
#         print(f'{matrix} : {similarity_matrix[pos]}')


# # most_similar(0, pairwise_similarities, 'Cosine Similarity')
# # most_similar(0, pairwise_differences, 'Euclidean Distance')


# # BERT ------------------------------


# # Verifique se a GPU está disponível
# # if torch.cuda.is_available():
# #     # Se estiver, defina o dispositivo como "cuda"
# #     device = torch.cuda.current_device()
# # else:
# #     # Caso contrário, defina o dispositivo como "cpu"
# #     device = 'cpu'

# # print('Encoding documents with device: ', device)

# anime_vectors_path = os.path.abspath(os.path.join(
#     os.path.dirname(__file__), '..', 'public', 'bert.anime.embeddings.dat'))

# sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

# # t = Timer()
# # t.start()

# # document_embeddings = sbert_model.encode(
# #     documents_df['documents_cleaned'], show_progress_bar=True, convert_to_numpy=True) # convert to numpy array

# # t.stop()

# # np.save('bert.anime.embeddings.npy', document_embeddings)

# # document_embeddings.dump(anime_vectors_path) # save to file


# text = "two brothers enter army to become alchemist"
# query_embedding = sbert_model.encode(
#     text, show_progress_bar=True, convert_to_numpy=True).reshape(1, -1)

# document_embeddings = np.load(anime_vectors_path, allow_pickle=True)

# pairwise_similarities = cosine_similarity(query_embedding, document_embeddings)
# pairwise_differences = euclidean_distances(
#     query_embedding, document_embeddings)

# most_similar(text, pairwise_similarities.flatten(), 'Cosine Similarity')
# most_similar(text, pairwise_differences.flatten(), 'Euclidean Distance')
