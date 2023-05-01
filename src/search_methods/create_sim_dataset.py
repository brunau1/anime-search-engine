import re
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from w2v_ranking import WordToVecRanking
from services.preprocess import read_animes_json

animes_file_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'public', 'dataset', 'animes.json'))

train_set_size = 1000
anime_data = read_animes_json(animes_file_path)

# text_vectors = WordToVecRanking(
#    anime_data[0], anime_data[1]).text_vectors  # use this for all dataset

text_vectors = WordToVecRanking(
     anime_data[0][:train_set_size], anime_data[1][:train_set_size]).text_vectors

higger_cosine_similarities = []
mean_cosine_similarities = []
lower_cosine_similarities = []

# Calcula a similaridade entre o texto atual e todos os demais
for i, _ in tqdm(enumerate(text_vectors)):
    curr_text = re.sub(r'[^a-z ]+', '', anime_data[1][i].lower())

    if len(curr_text.split()) > 64:
        sim = cosine_similarity([text_vectors[i]], text_vectors).flatten()
        most_similar_idx = sim.argsort()[
            ::-1][1]
        mean_similar_idx = sim.argsort()[::-1][round(len(sim)//7, 0)]

        lower_similar_idx = sim.argsort()[::-1][round(len(sim)//4, 0)]

        if sim[most_similar_idx] < 0.92:
            higger_cosine_similarities.append(
                [i, most_similar_idx, round(sim[most_similar_idx], 4)])

        mean_cosine_similarities.append(
            [i, mean_similar_idx, round(sim[mean_similar_idx], 4)])

        lower_cosine_similarities.append(
            [i, lower_similar_idx, round(sim[lower_similar_idx], 4)])

        # print('Text: ', i, 'Most similar: ', most_similar_idx,
        #       'Similarity: ', sim[most_similar_idx])


higger_sim_values = np.argsort([score[2]
                               for score in higger_cosine_similarities])[::-1]

higger_mean_sim_values = np.argsort([score[2]
                                    for score in mean_cosine_similarities])[::-1]


# print('Higger cosine similarities: ', higger_cosine_similarities[:5])
# print('Mean cosine similarities: ', mean_cosine_similarities[:5])

final_similarities = []

curr_len = round(len(higger_cosine_similarities)//2, 0)

for i in range(0, curr_len):
    final_similarities.append(
        higger_cosine_similarities[higger_sim_values[i]])

print("Curr len higger: ", curr_len)

curr_len = round(len(mean_cosine_similarities)//4, 0)
for i in range(0, curr_len):
    final_similarities.append(
        mean_cosine_similarities[higger_mean_sim_values[i]])

print("Curr len mean: ", curr_len)

curr_len = round(len(lower_cosine_similarities)//8, 0)
for i in range(0, curr_len):
    final_similarities.append(
        lower_cosine_similarities[i])

print("Curr len lower: ", curr_len)

print('Final similarities: ', len(final_similarities))
# print('Final similarities: ', final_similarities)

public_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'public'))

np.save(os.path.join(public_path, 'dataset',
                     f"similarities_{train_set_size}.npy"), final_similarities)

# loaded = np.load(os.path.join(public_path, 'dataset', f"similarities_{train_set_size}.npy"))

# print('Similarities saved shape: ', loaded.shape)
