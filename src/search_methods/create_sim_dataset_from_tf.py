import tensorflow_hub as hub
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from services.preprocess import read_animes_json, bert_simple_preprocess_text, simple_preprocess_text

os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# physical_devices = tf.config.list_physical_devices('GPU')

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#     try:
#         tf.config.set_logical_device_configuration(
#             gpus[0],
#             [tf.config.LogicalDeviceConfiguration(memory_limit=2048)])
#         logical_gpus = tf.config.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Virtual devices must be set before GPUs have been initialized
#         print(e)


module_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'public', 'dataset', 'modules', 'universal-sentence-encoder-large_5'))

animes_file_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'public', 'dataset', 'animes.json'))

anime_data = read_animes_json(animes_file_path)

# gerar embeddings para todos os textos e salvar em um arquivo
# set_size = 5

# texts = [bert_simple_preprocess_text(text)
#          for text in anime_data[1][:]]

# texts_embeddings = []

# with tf.device("GPU:0"):
#     model = hub.load(module_path)

#     for i, text in tqdm(enumerate(texts)):
#         texts_embeddings.append(model([text]).numpy()[0])


# print(np.array(texts_embeddings).shape)

embeddings_file_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'public', 'dataset', 'modules', 'sts_large_anime_embeddings.npy'))

# np.save(embeddings_file_path, texts_embeddings)

# # gerar matriz de similaridade entre todos os textos

search_phrases = ["the soldiers fight to protect the goddess athena",
                  "the protagonist is a demon who wants to become a hero",
                  "the volleyball team is the main focus of the anime",
                  "a man who can defeat any enemy with one punch",
                  "it has a dragon wich give three wishes to the one who find it",
                  "two brothers enter army to become alchemist",
                  "a ninja boy who wants to become a hokage",
                  "a boy who wants to become the pirate king",
                  "the philosopher stone grants immortality to the one who find it",
                  "the protagonist got the shinigami sword and now he has to kill hollows",
                  "give me an anime about giant robots"]

texts_embeddings = np.load(embeddings_file_path)

model = hub.load(module_path)

print("model loaded\n")

NUM_EXAMPLES = [100, 500, 1000, 2000, 5000, 10000, len(texts_embeddings)]

all_similarities = []
for i, qtd_texts in tqdm(enumerate(NUM_EXAMPLES)):
    print(f"qtd_texts: {qtd_texts}")

    curr_texts_embeddings = np.array(texts_embeddings[:qtd_texts])

    print("curr_texts_embeddings shape: ", curr_texts_embeddings.shape)

    curr_all_similarities = []

    for search_text in search_phrases:
        search_encoded = model(
            [bert_simple_preprocess_text(search_text)]).numpy()[0]

        # print("search phrase: ", search_text, "\n")

        similarity_matrix = np.inner(search_encoded, curr_texts_embeddings)

        # print(similarity_matrix)

        similar_sentence_indices = np.argsort(similarity_matrix)[::-1]

        curr_similarities = []
        for i in range(0, 10):
            # print(similar_sentence_indices[i],
            #       anime_data[0][similar_sentence_indices[i]])
            curr_similarities.append(
                [similar_sentence_indices[i], anime_data[0][similar_sentence_indices[i]], similarity_matrix[similar_sentence_indices[i]]])

        curr_all_similarities.append(f"'{search_text}' -> {curr_similarities}")

    all_similarities.append(curr_all_similarities)


for i, gen_similarities in enumerate(all_similarities):
    results_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'public', 'dataset', 'search_results', f"sts_sim_results_{NUM_EXAMPLES[i]}.txt"))

    with open(results_path, 'w', encoding='utf-8') as f:
        for line in gen_similarities:
            f.write(f"{line}\n")


# highest_similarities = []
# mean_similarities = []
# lower_similarities = []

# for i, text_embed in enumerate(texts_embeddings):
#     similar_sentence_indices = np.argsort(similarity_matrix[i])[::-1]

#     most_similar_idx = similar_sentence_indices[1]

#     mean_similar_idx = similar_sentence_indices[round(
#         len(similar_sentence_indices)//2, 0)]

#     lower_similar_idx = similar_sentence_indices[len(
#         similar_sentence_indices)-1]

#     highest_similarities.append(
#         [i, most_similar_idx, similarity_matrix[i][most_similar_idx]])

#     mean_similarities.append(
#         [i, mean_similar_idx, similarity_matrix[i][mean_similar_idx]])

#     lower_similarities.append(
#         [i, lower_similar_idx, similarity_matrix[i][lower_similar_idx]])


# para cada conjunto de similaridades, deve remover os pares de indices duplicados

# for i, _ in enumerate(highest_similarities):
#     if i != highest_similarities[i][1]:
#         highest_similarities.remove(highest_similarities[i])

# for i, _ in enumerate(mean_similarities):
#     if i != mean_similarities[i][1]:
#         mean_similarities.remove(mean_similarities[i])

# for i, _ in enumerate(lower_similarities):
#     if i != lower_similarities[i][1]:
#         lower_similarities.remove(lower_similarities[i])

# print("highest_similarities: ", len(highest_similarities))
# print("mean_similarities: ", len(mean_similarities))
# print("lower_similarities: ", len(lower_similarities))
