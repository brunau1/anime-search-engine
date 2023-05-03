import os
import time
import numpy as np
import tensorflow_hub as hub

from services.preprocess import simple_preprocess_text, read_animes_json, bert_simple_preprocess_text
from services.ranking import cos_similarity_top_results, euclidean_distance_top_results, calculate_bleu_1_score_for_texts

dataset_public_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'public', 'dataset'))

# texts_embeddings_file_path = os.path.abspath(os.path.join(
#     dataset_public_path, 'modules', 'sts_large_anime_embeddings.npy'))

texts_embeddings_file_path = os.path.abspath(os.path.join(
    dataset_public_path, 'modules', 'sts_large_anime_embeddings_no_stop_words.npy'))

module_path = os.path.abspath(os.path.join(
    dataset_public_path, 'modules', 'universal-sentence-encoder-large_5'))


class STSBertRanking:
    def __init__(self, titles, synopsis, set_size=10000):
        self.titles = titles[:set_size]

        self.synopsis = [' '.join(simple_preprocess_text(text))
                         for text in synopsis[:set_size]]
        self.synopsis_embeddings = np.load(texts_embeddings_file_path)
        self.synopsis_embeddings = self.synopsis_embeddings[:set_size]

        print("Bert texts embeddings loaded!")
        print("Loading STS BERT model...")

        self.model = hub.load(module_path)

        print("STS BERT model loaded!")

    def search(self, query, similarity_method="cosine", top_k=10):
        t = time.time()
        query = ' '.join(simple_preprocess_text(query))
        query_embedding = self.model([query]).numpy()[0]

        ranking = []
        if similarity_method == "cosine":
            ranking = cos_similarity_top_results(
                query_embedding.reshape(1, -1), self.synopsis_embeddings, self.titles, top_k)

            ranking = [[r[0], np.float64(r[1])] for r in ranking]

            print("search time: ", time.time() - t)
            return ranking

        elif similarity_method == "euclidean":
            ranking = euclidean_distance_top_results(
                query_embedding, self.synopsis_embeddings, self.titles, top_k)

            print("search time: ", time.time() - t)
            return ranking

    def search_with_bleu(self, query, top_k=10):
        t = time.time()
        query = ' '.join(simple_preprocess_text(query))
        query_embedding = self.model([query]).numpy()[0]

        ranking = calculate_bleu_1_score_for_texts(
            self.titles, self.synopsis, query, query_embedding, self.synopsis_embeddings, top_k)

        print("search time: ", time.time() - t)
        return ranking


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


def main():
    animes_file_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'public', 'dataset', 'animes.json'))

    anime_data = read_animes_json(animes_file_path)

    model = STSBertRanking(anime_data[0], anime_data[1])

    lines = []
    for search_text in search_phrases:
        print("search phrase: ", search_text, "\n")

        ranking = model.search_with_bleu(search_text)

        lines.append(f"'{search_text}' --> {ranking}\n")

    out_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'public', 'dataset', 'search_results', 'sts_bert_no_stop_10000.txt'))

    with open(out_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line)


main()
