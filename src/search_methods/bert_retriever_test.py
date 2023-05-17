from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
import re
import json
import torch
import numpy as np
import pandas as pd

from preprocess import bert_simple_preprocess_text
from ranking import calculate_bleu_1_score_for_texts

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

print(f"Using {DEVICE} device")


bert_trained_model_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'public', 'models', 'bert_trained_model__v5'))

model = SentenceTransformer(bert_trained_model_path)

model.to(DEVICE)

animes_file_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'public', 'dataset', 'animes.json'))

with open(animes_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

titles = data['names']  # array com os títulos dos textos
synopses = data['content'][:]  # array com os textos


def find_similar(vector_representation, all_representations, k=1):
    similarity_matrix = cosine_similarity(
        vector_representation, all_representations)
    # np.fill_diagonal(similarity_matrix, 0)
    similarities = similarity_matrix[0]
    if k == 1:
        return [np.argmax(similarities)]
    elif k is not None:
        return np.flip(similarities.argsort()[-k:][::1])


cleaned_synopses = [bert_simple_preprocess_text(synopsis)
                    for synopsis in synopses]

sinopses_embeddings = model.encode(
    cleaned_synopses, show_progress_bar=True)

print("sinopses_embeddings shape: ", sinopses_embeddings.shape)

# search_phrases = ["the soldiers fight to protect the goddess athena",
#                   "the protagonist is a demon who wants to become a hero",
#                   "the protagonist gains the power to kill anyone whose name he writes in a notebook",
#                   "a boy was possessed by a demon and now he has to fight demons",
#                   "the volleyball team is the main focus of the anime",
#                   "the anime shows the daily life of a volleyball team in high school",
#                   "a man who can defeat any enemy with one punch",
#                   "the protagonist become skinny just training",
#                   "it has a dragon wich give three wishes to the one who find it",
#                   "the protagonist always use the wishes to revive his friends",
#                   "the philosopher stone grants immortality to the one who find it",
#                   "two brothers lost their bodies and now they have to find the philosopher stone",
#                   "a ninja kid who wants to become a hokage",
#                   "the protagonist's dream is to become the pirate king",
#                   "the protagonist uses a straw hat and can stretch his body",
#                   "the protagonist got the shinigami sword and now he has to kill hollows",
#                   "it was a knight who use a saint armor blessed by the goddess athena",
#                   "the protagonist met a shinigami and goes to the soul society"]

search_phrases = ["the protagonist gains the power to kill anyone whose name he writes in a notebook",
                  "a man who can defeat any enemy with one punch",
                  "the anime shows a volleyball team which trains to become the best of japan",
                  "the protagonist has the power of stretch his body and use a straw hat",
                  "the sayan warrior revive his friends using the wish given by the dragon",
                  "the philosopher stone grants power and immortality to the one who find it",
                  "two brothers lost their bodies and now they have to find the philosopher stone",
                  "a ninja kid who wants to become the best ninja of his village and has a demon inside him",
                  "the protagonist got the shinigami powers and now he has to kill hollows",
                  "it was a knight who use a saint armor blessed by the goddess athena"]


def search(query_text):
    query_embedding = model.encode([bert_simple_preprocess_text(query_text)])

    distilbert_similar_indexes = find_similar(
        query_embedding, sinopses_embeddings, 10)

    results = calculate_bleu_1_score_for_texts(
        titles, synopses, query_text, query_embedding, sinopses_embeddings)

    output_data = []
    for index in distilbert_similar_indexes:
        output_data.append(titles[index])

    print("output_data: ", output_data)
    return results


lines = []
for search_text in search_phrases:
    print("search phrase: ", search_text, "\n")

    results = search(search_text)

    lines.append(f"'{search_text}' --> {results}\n")

out_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'public', 'dataset', 'search_results', 'STS_bert_15k.txt'))

with open(out_path, 'w', encoding='utf-8') as f:
    for line in lines:
        f.write(line)
