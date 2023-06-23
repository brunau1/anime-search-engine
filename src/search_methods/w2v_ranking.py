import os
import json
from tqdm import tqdm
import numpy as np
from gensim.models import Word2Vec, KeyedVectors

from preprocess import preprocess_text, read_animes_json, preprocess_text
from ranking import cos_similarity_top_results, euclidean_distance_top_results, calculate_bleu_1_score_for_texts
from timer import Timer
from evaluate_models import calculate_metrics, calculate_mean_average_model_evaluation_metrics, calculate_bleu_score

# from src.search_methods.services.timer import Timer
# from src.search_methods.services.ranking import cos_similarity_top_results, euclidean_distance_top_results, calculate_bleu_1_score_for_texts
# from src.search_methods.services.preprocess import preprocess_text

VECTOR_SIZE = 200


def train_model(vector_size=200):
    # Carrega os dados
    animes_file_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'public', 'dataset', 'animes.json'))

    with open(animes_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    content = data['content']  # array com os textos

    # Aplica o pré-processamento nos textos
    processed_content = [preprocess_text(
        text) for text in content if len(text.split()) > 40]

    print('Data loaded. Number of texts: ', len(processed_content))
    model = Word2Vec(min_count=1,  # 3
                     window=3,  # 5
                     sample=6e-5,  # 1e-5
                     vector_size=vector_size,
                     alpha=0.03,
                     min_alpha=0.007,
                     workers=4)

    model.build_vocab(processed_content)

    print('Training model...')
    t = Timer()
    t.start()

    model.train(processed_content, total_examples=len(
        processed_content), epochs=300)

    t.stop()

    w2v_model_file_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'public', 'models', 'word2vec.model'))

    model.save(w2v_model_file_path)

    print('Model created. Shape: ', model.wv.vectors.shape)


def build_text_vectors(processed_content, model, vector_size):
    # Converte cada texto para um vetor Word2Vec
    print('Building text vectors...')
    vectors = []
    for text in processed_content:
        text_vector = np.zeros(vector_size)
        for token in text:
            if token in model.wv:
                text_vector = np.add(text_vector, model.wv[token])

        vectors.append(text_vector)

    print('Text vectors created. Shape: ', np.array(vectors).shape)
    return vectors

# Define a função de busca


def search(query, names, text_vectors, model, vector_size, top_k=10, similarity_method='cosine'):
    # t = Timer()
    # t.start()
    # print('Searching for: "', query) #, '" using',
    #   similarity_method, 'similarity')
    # Pré-processa a consulta
    processed_query = preprocess_text(query)

    print('processed_query: ', processed_query)

    query_vector = np.zeros(vector_size)
    for token in processed_query:
        if token in model.wv:
            query_vector = np.add(query_vector, model.wv[token])

    ranking = []

    if similarity_method == 'cosine':
        query_vector = query_vector.reshape(1, -1)

        ranking = cos_similarity_top_results(
            query_vector, text_vectors, names, top_k)

    elif similarity_method == 'euclidean':
        ranking = euclidean_distance_top_results(
            query_vector, text_vectors, names, top_k)

    # t.stop()
    return ranking


def search_and_bleu(query, names, texts, text_vectors, model, vector_size, top_k=10):
    t = Timer()
    t.start()
    print('Searching for: "', query, '"')
    processed_query = preprocess_text(query)

    query_vector = np.zeros(vector_size)
    for token in processed_query:
        if token in model.wv:
            query_vector = np.add(query_vector, model.wv[token])

    query_vector = query_vector.reshape(1, -1)

    texts = [' '.join(text) for text in texts]

    ranking = calculate_bleu_1_score_for_texts(
        names, texts, query, query_vector, text_vectors, top_k)

    t.stop()
    return ranking


class WordToVecRanking:
    def __init__(self, names, sinopsis):
        self.names = names  # array com os títulos dos textos

        self.processed_content = [
            preprocess_text(text) for text in sinopsis]

        # Define o tamanho dos vetores de saída do modelo Word2Vec
        self.vector_size = VECTOR_SIZE

        w2v_model_file_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', 'public', 'models', 'word2vec.model'))

        print('Loading model from: ', w2v_model_file_path)
        self.model = KeyedVectors.load(w2v_model_file_path)

        self.text_vectors = build_text_vectors(
            self.processed_content, self.model, self.vector_size)

    def search(self, query, similarity_method, top_k=10):
        return search(query, self.names, self.text_vectors, self.model, self.vector_size, top_k, similarity_method)

    def search_and_bleu(self, query, top_k=10):
        return search_and_bleu(query, self.names, self.processed_content, self.text_vectors, self.model, self.vector_size, top_k)


def calculate_f_score(relevant_docs, retrieved_docs):
    precision = len(set(relevant_docs).intersection(
        set(retrieved_docs))) / float(len(retrieved_docs))
    recall = len(set(relevant_docs).intersection(
        set(retrieved_docs))) / float(len(relevant_docs))
    if precision + recall == 0:
        return 0
    f_score = 2 * (precision * recall) / (precision + recall)
    return f_score


# usage example
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


def main():
    animes_file_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'public', 'dataset', 'animes_with_cover.json'))

    eval_data_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'public', 'dataset', 'sentences-and-related-docs.json'))

    with open(eval_data_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    anime_data = read_animes_json(animes_file_path)

    model = WordToVecRanking(anime_data[0], anime_data[1])

    evaluated_metrics = []

    print("Evaluating...", len(eval_data))

    for k in [1, 3, 5, 7, 10]:
        current_evaluated_metrics = {
            "precision": [],
            "recall": [],
            "f_score": [],
            "f1_score_binary": [],
            "f1_score_macro": [],
            "f1_score_micro": [],
            "f1_score_weighted": [],
            "bleu": []
        }

        print("k: ", k)

        for _, curr_eval_data in tqdm(enumerate(eval_data)):
            print("title: ", curr_eval_data['title'])

            curr_queries = curr_eval_data['sentences']

            for curr_query in curr_queries:
                print("curr_query: ", curr_query)

                relevant_doc_ids = curr_eval_data['relatedDocs']

                top_idxes = model.search(curr_query, 'cosine', k)

                print("top_idxes: ", top_idxes)
                print("relevant_doc_ids: ", relevant_doc_ids)

                metrics = calculate_metrics(relevant_doc_ids, top_idxes, k)

                for _, idx in enumerate(top_idxes):
                    current_evaluated_metrics['bleu'].append(
                        calculate_bleu_score(curr_query, anime_data[1], idx))

                current_evaluated_metrics['precision'].append(
                    metrics['precision'])
                current_evaluated_metrics['recall'].append(metrics['recall'])
                current_evaluated_metrics['f_score'].append(metrics['f_score'])
                current_evaluated_metrics['f1_score_binary'].append(
                    metrics['f1_score_binary'])
                current_evaluated_metrics['f1_score_macro'].append(
                    metrics['f1_score_macro'])
                current_evaluated_metrics['f1_score_micro'].append(
                    metrics['f1_score_micro'])
                current_evaluated_metrics['f1_score_weighted'].append(
                    metrics['f1_score_weighted'])

        average_model_metrics = calculate_mean_average_model_evaluation_metrics(
            current_evaluated_metrics)

        print("average_model_metrics: ", average_model_metrics)

        evaluated_metrics.append({
            "k": k,
            "average_metrics": average_model_metrics,
            "all_metrics": current_evaluated_metrics
        })

    out_path_all = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'public', 'dataset', 'search_results', 'w2v_evaluation_metrics_all_15k.json'))

    with open(out_path_all, 'w', encoding='utf-8') as f:
        json.dump(evaluated_metrics, f, indent=4)

    # calculated_f_scores = []

    # print("Evaluating...", len(eval_data))

    # for _, data in tqdm(enumerate(eval_data)):
    #     curr_query = data['query']
    #     relevant_doc_id = data['relevant_doc_id']

    #     top_idx = model.search(curr_query, 'cosine')

    #     calculated_f_scores.append(
    #         calculate_f_score([relevant_doc_id], [top_idx]))

    # print('F-Score: ', np.mean(calculated_f_scores))

    # # save results

    # out_path = os.path.abspath(os.path.join(
    #     os.path.dirname(__file__), '..', '..', 'public', 'dataset', 'search_results', 'w2v_cosine_15k.txt'))

    # with open(out_path, 'w', encoding='utf-8') as f:
    #     f.write(f'F-Score: {np.mean(calculated_f_scores)}\n')
    #     f.write(f'F-Scores: {calculated_f_scores}\n')

    # lines = []
    # for search_text in search_phrases:
    #     print("search phrase: ", search_text, "\n")

    #     # ranking = model.search_and_bleu(search_text)
    #     top_index = model.search(search_text, 'cosine')

    #     lines.append(f"'{search_text}' --> {ranking}\n")

    # out_path = os.path.abspath(os.path.join(
    #     os.path.dirname(__file__), '..', '..', 'public', 'dataset', 'search_results', 'w2v_bleu_15k.txt'))

    # with open(out_path, 'w', encoding='utf-8') as f:
    #     for line in lines:
    #         f.write(line)


# train_model()
main()
