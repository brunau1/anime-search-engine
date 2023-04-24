from preprocess import read_animes_json
from w2v_ranking import WordToVecRanking
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import os
import re


anime_data = read_animes_json()
s_query = 'two brothers enter army to become alchemist'
# ranking = WordToVecRanking(
#     anime_data[0], anime_data[1])
# print(ranking)
text_vectors = WordToVecRanking(
    anime_data[0][:1000], anime_data[1][:1000]).text_vectors

higger_cosine_similarities = []
mean_cosine_similarities = []
# # Calcula a similaridade entre o texto atual e todos os demais
for i, _ in tqdm(enumerate(text_vectors)):
    curr_text = re.sub(r'[^a-z ]+', '', anime_data[1][i].lower())

    if len(curr_text.split()) > 96:
        sim = cosine_similarity([text_vectors[i]], text_vectors).flatten()
        most_similar_idx = sim.argsort()[
            ::-1][1]
        mean_similar_idx = sim.argsort()[::-1][len(sim)//4]

        if sim[most_similar_idx] < 0.92:
            higger_cosine_similarities.append(
                [i, most_similar_idx, round(sim[most_similar_idx], 4)])

        mean_cosine_similarities.append(
            [i, mean_similar_idx, round(sim[mean_similar_idx], 4)])

        # print('Text: ', i, 'Most similar: ', most_similar_idx,
        #       'Similarity: ', sim[most_similar_idx])

higger_sim_values = np.argsort([score[2]
                               for score in higger_cosine_similarities])[::-1]

higger_mean_sim_values = np.argsort([score[2]
                                    for score in mean_cosine_similarities])[::-1]

# print('Higger cosine similarities: ', higger_cosine_similarities[:5])
# print('Mean cosine similarities: ', mean_cosine_similarities[:5])
# print('Higger sim values len: ', len(higger_sim_values))
# print('similar texts: ', anime_data[1][708], "\nthe second -----: ", anime_data[1][719])

final_similarities = []

for i, _ in enumerate(higger_cosine_similarities):
    if i % 2 == 0:
        final_similarities.append(
            higger_cosine_similarities[higger_sim_values[i]])
    elif i % 3 == 0:
        final_similarities.append(
            mean_cosine_similarities[higger_mean_sim_values[i]])
    else:
        final_similarities.append(
            mean_cosine_similarities[i])

print('Final similarities: ', len(final_similarities))
print('Final similarities ex: ', final_similarities[:10])

public_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'public'))

np.save(os.path.join(public_path, 'dataset',
        'similarities.npy'), final_similarities)

loaded = np.load(os.path.join(public_path, 'dataset', 'similarities.npy'))

print('Similarities saved shape: ', loaded.shape)
