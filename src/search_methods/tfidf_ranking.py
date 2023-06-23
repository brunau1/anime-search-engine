import os
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from tqdm import tqdm
import numpy as np

from timer import Timer
from preprocess import preprocess_text, read_animes_json
from ranking import cos_similarity_top_results, euclidean_distance_top_results, calculate_bleu_1_score_for_texts
from evaluate_models import calculate_metrics, calculate_mean_average_model_evaluation_metrics, calculate_bleu_score

# from src.search_methods.services.timer import Timer
# from src.search_methods.services.preprocess import preprocess_text, read_animes_json
# from src.search_methods.services.ranking import cos_similarity_top_results, euclidean_distance_top_results, calculate_bleu_1_score_for_texts
# Carrega os dados do arquivo JSON e faz o pré-processamento


def load_tfidf_vectors(names, processed_content):
    print('Loading tfidf data...')
    # Pré-processa as sinopses
    processed_texts = [' '.join(text) for text in processed_content]

    # Calcula os vetores TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    text_vectors = tfidf_matrix.toarray()

    print('Data loaded. TF-IDF matrix shape: ', tfidf_matrix.shape, '\n')
    # Armazena os dados em uma estrutura de dados para ser usada na busca
    return {'titles': names, 'sinopsis': processed_texts, 'tfidf_matrix': tfidf_matrix, 'vectorizer': vectorizer, 'raw_text_vectors': text_vectors}


# Função que realiza a busca
def get_similarity_ranking(query, anime_data, rank_count=10, similarity_method='cosine'):
    # t = Timer()
    # t.start()
    # print('Searching for: "', query, '" using',
    #       similarity_method, 'similarity')
    # Pré-processa a consulta
    query = ' '.join(preprocess_text(query))
    # Transforma a consulta em vetor TF-IDF
    query_vector = anime_data['vectorizer'].transform([query])

    ranking = []
    if similarity_method == 'cosine':
        ranking = cos_similarity_top_results(
            query_vector, anime_data['tfidf_matrix'], anime_data['titles'], rank_count)

    elif similarity_method == 'euclidean':
        # vetores redimensionados a 1D para serem
        # usados no calculo de distância euclidiana
        query_vector = query_vector.toarray()[0]
        text_vectors = anime_data['raw_text_vectors']

        ranking = euclidean_distance_top_results(
            query_vector, text_vectors, anime_data['titles'], rank_count)

    # t.stop()
    return ranking


def get_bleu_ranking(query, anime_data, rank_count=10):
    t = Timer()
    t.start()
    print('Searching for: ', query)
    # Pré-processa a consulta
    query = ' '.join(preprocess_text(query))
    # Transforma a consulta em vetor TF-IDF
    query_vector = anime_data['vectorizer'].transform([query])

    ranking = calculate_bleu_1_score_for_texts(
        anime_data['titles'], anime_data['sinopsis'], query, query_vector, anime_data['raw_text_vectors'], rank_count)

    t.stop()
    return ranking


class TfIdfRanking:
    def __init__(self, names, sinopsis):
        processed_content = [preprocess_text(text) for text in sinopsis]
        self.anime_data = load_tfidf_vectors(names, processed_content)

    def search(self, query, similarity_method, rank_count=10):
        return get_similarity_ranking(query, self.anime_data, rank_count, similarity_method)

    def search_bleu(self, query, rank_count=10):
        return get_bleu_ranking(query, self.anime_data, rank_count)


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
search_phrases = [
    "a man who can defeat any enemy with one punch"]


def main():
    animes_file_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'public', 'dataset', 'animes_with_cover.json'))

    anime_data = read_animes_json(animes_file_path)

    eval_data_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'public', 'dataset', 'sentences-and-related-docs.json'))

    with open(eval_data_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    model = TfIdfRanking(anime_data[0], anime_data[1])

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
        os.path.dirname(__file__), '..', '..', 'public', 'dataset', 'search_results', 'tfidf_evaluation_metrics_all_15k.json'))

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
    #     os.path.dirname(__file__), '..', '..', 'public', 'dataset', 'search_results', 'tfidf_cosine_15k.txt'))

    # with open(out_path, 'w', encoding='utf-8') as f:
    #     f.write(f'F-Score: {np.mean(calculated_f_scores)}\n')
    #     f.write(f'F-Scores: {calculated_f_scores}\n')

    # lines = []
    # for search_text in search_phrases:
    #     print("search phrase: ", search_text, "\n")

    #     ranking = model.search_bleu(search_text)
    #     # ranking = model.search(search_text, 'cosine')

    #     lines.append(f"'{search_text}' --> \n{ranking}\n")

    # out_path = os.path.abspath(os.path.join(
    #     os.path.dirname(__file__), '..', '..', 'public', 'dataset', 'search_results', 'tf_idf_bleu_15k.txt'))

    # with open(out_path, 'w', encoding='utf-8') as f:
    #     for line in lines:
    #         f.write(line)


main()
