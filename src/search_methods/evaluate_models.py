import os
import numpy as np
import json
from sklearn.metrics import f1_score

from preprocess import preprocess_text


evaluation_data_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'public', 'dataset', 'sentences-and-related-docs.json'))

with open(evaluation_data_path, 'r', encoding='utf-8') as f:
    evaluation_data = json.load(f)


def calculate_f_score(relevant_docs, retrieved_docs):
    precision = len(set(relevant_docs).intersection(
        set(retrieved_docs))) / float(len(retrieved_docs))
    recall = len(set(relevant_docs).intersection(
        set(retrieved_docs))) / float(len(relevant_docs))
    if precision + recall == 0:
        return 0
    f_score = 2 * (precision * recall) / (precision + recall)
    return f_score


def calculate_f1_score_binary(true_labels, pred_labels):
    return f1_score(true_labels, pred_labels, average='binary')


def calculate_f1_score_macro(relevant_docs, retrieved_docs):
    return f1_score(relevant_docs, retrieved_docs, average='macro')


def calculate_f1_score_micro(relevant_docs, retrieved_docs):
    return f1_score(relevant_docs, retrieved_docs, average='micro')


def calculate_f1_score_weighted(relevant_docs, retrieved_docs):
    return f1_score(relevant_docs, retrieved_docs, average='weighted')


def calculate_precision(relevant_docs, retrieved_docs):
    return len(set(relevant_docs).intersection(set(retrieved_docs))) / float(len(retrieved_docs))


def calculate_recall(relevant_docs, retrieved_docs):
    return len(set(relevant_docs).intersection(set(retrieved_docs))) / float(len(relevant_docs))


def get_true_pred_binary_labels(relevant_docs, retrieved_docs):
    true_labels = []
    pred_labels = []

    # for each value in relevant_docs append 1 to true_labels
    # and fill to pred_labels length with 0

    # for each value in retrieved_docs wich appear in relevant_docs
    # append 1 to pred_labels and 0 if not

    for doc in retrieved_docs:
        if doc in relevant_docs:
            pred_labels.append(1)
        else:
            pred_labels.append(0)

    for doc in relevant_docs:
        true_labels.append(1)

    # fill true_labels with 0 to pred_labels length
    true_labels += [0] * (len(pred_labels) - len(true_labels))

    return true_labels, pred_labels


def get_true_pred_macro_labels(relevant_docs, retrieved_docs):
    true_labels = []
    pred_labels = []

    # for each value in relevant_docs asign an unique id to true_labels equal to 1 and i++ for the next values
    # and create a second array with the same length as relevant_docs with the true ids of each doc in the same order of the
    # asignation of the crescent ids

    curr_relevant_docs = []
    curr_real_relevant_ids = []
    for i, doc in enumerate(relevant_docs):
        curr_relevant_docs.append(i+1)
        curr_real_relevant_ids.append(doc)

    for doc in retrieved_docs:
        if doc in curr_real_relevant_ids:
            # append the value of the corresponding index of the doc in curr_relevant_docs
            pred_labels.append(
                curr_relevant_docs[curr_real_relevant_ids.index(doc)])
        else:
            pred_labels.append(0)

    for doc in curr_relevant_docs:
        true_labels.append(doc)

    # fill true_labels with 0 to pred_labels length
    true_labels += [0] * (len(pred_labels) - len(true_labels))

    # output example:
    # relevant_docs = [1, 2, 3, 4, 5]
    # retrieved_docs = [0, 1, 0, 4, 0]
    return true_labels, pred_labels


'''
- foreach saved result for metrics calculated for each model
- calculate the average of each metric and return a dict with the results
'''


def calculate_average_model_evaluation_metrics(model_metrics):
    average_metrics = {
        "precision": 0,
        "recall": 0,
        "f_score": 0,
        "f1_score_binary": 0,
        "f1_score_macro": 0,
        "f1_score_micro": 0,
        "f1_score_weighted": 0
    }

    for metric in average_metrics:
        average_metrics[metric] += np.array(model_metrics[metric]).sum()
        average_metrics[metric] /= len(model_metrics)

    return average_metrics


def calculate_mean_average_model_evaluation_metrics(model_metrics):
    average_metrics = {
        "precision": 0,
        "recall": 0,
        "f_score": 0,
        "f1_score_binary": 0,
        "f1_score_macro": 0,
        "f1_score_micro": 0,
        "f1_score_weighted": 0
    }

    for metric in average_metrics:
        average_metrics[metric] += np.array(model_metrics[metric]).mean()

    return average_metrics

# for each metric should compare the result of each model and return percentage of improvement


def compare_models_evolution_based_on_metrics(model1_metrics, model2_metrics):
    improvement_metrics = {
        "precision": 0,
        "recall": 0,
        "f_score": 0,
        "f1_score_binary": 0,
        "f1_score_macro": 0,
        "f1_score_micro": 0,
        "f1_score_weighted": 0
    }

    for metric in improvement_metrics:
        higger = model2_metrics[metric]
        lower = model1_metrics[metric]

        # calculate percentage of improvement
        improvement_metrics[metric] = (
            abs(higger - lower) / ((higger + lower)/2)) * 100.0

    return improvement_metrics


def calculate_metrics(relevant_docs, retrieved_docs, K=10):
    eval_metrics = {
        "precision": 0,
        "recall": 0,
        "f_score": 0,
        "f1_score_binary": 0,
        "f1_score_macro": 0,
        "f1_score_micro": 0,
        "f1_score_weighted": 0
    }

    # pick each doc in relevant_docs and check if it is in retrieved_docs and append the ids in an separate array
    curr_relevant_docs = []
    for doc in relevant_docs:
        if doc in retrieved_docs:
            curr_relevant_docs.append(doc)

    # the relevant docs is the current relevant docs plus the rest of the relevant docs that are not in current relevant docs
    reference_docs = curr_relevant_docs + \
        list(set(relevant_docs) - set(curr_relevant_docs))

    # now pick the K first docs in relevant_docs
    reference_docs = reference_docs[:K]

    binary_true_labels, binary_pred_labels = get_true_pred_binary_labels(
        reference_docs, retrieved_docs)

    macro_true_labels, macro_pred_labels = get_true_pred_macro_labels(
        reference_docs, retrieved_docs)

    print(f"binary true labels: {binary_true_labels}")
    print(f"binary pred labels: {binary_pred_labels}")
    print(f"macro true labels: {macro_true_labels}")
    print(f"macro pred labels: {macro_pred_labels}")

    eval_metrics["precision"] = calculate_precision(
        reference_docs, retrieved_docs)

    eval_metrics["recall"] = calculate_recall(reference_docs, retrieved_docs)

    eval_metrics["f_score"] = calculate_f_score(reference_docs, retrieved_docs)

    eval_metrics["f1_score_binary"] = calculate_f1_score_binary(
        binary_true_labels, binary_pred_labels)

    eval_metrics["f1_score_macro"] = calculate_f1_score_macro(
        macro_true_labels, macro_pred_labels)

    eval_metrics["f1_score_micro"] = calculate_f1_score_micro(
        macro_true_labels, macro_pred_labels)

    eval_metrics["f1_score_weighted"] = calculate_f1_score_weighted(
        macro_true_labels, macro_pred_labels)

    # print(
    #     f"precision: {calculate_precision(reference_docs, retrieved_docs[i])}")
    # print(f"recall: {calculate_recall(reference_docs, retrieved_docs[i])}")
    # print(f"f-score: {calculate_f_score(reference_docs, retrieved_docs[i])}")
    # print(
    #     f"f1-score binary: {calculate_f1_score_binary(binary_true_labels, binary_pred_labels)}")
    # print(
    #     f"f1-score macro: {calculate_f1_score_macro(macro_true_labels, macro_pred_labels)}")
    # print(
    #     f"f1-score micro: {calculate_f1_score_micro(macro_true_labels, macro_pred_labels)}")
    # print(
    #     f"f1-score weighted: {calculate_f1_score_weighted(macro_true_labels, macro_pred_labels)}")
    print()

    return eval_metrics


# out_path = os.path.abspath(os.path.join(
#     os.path.dirname(__file__), '..', '..', 'public', 'dataset', 'evaluation'))

# K = 10
# lines = []

# print("\nWord2Vec ------------------")
# w2v_metrics = calculate_metrics(expected_results, w2v_results, K)
# lines.append(f"Word2Vec metrics--> {w2v_metrics}\n")

# w2v_average_metrics = calculate_average_model_evaluation_metrics(w2v_metrics)
# # lines.append(f"Word2Vec average metrics--> {w2v_average_metrics}\n")

# w2v_mean_average_metrics = calculate_mean_average_model_evaluation_metrics(
#     w2v_metrics)
# lines.append(f"Word2Vec mean average metrics--> {w2v_mean_average_metrics}\n")

# w2v_out_path = os.path.abspath(
#     os.path.join(out_path, f"w2v_metrics_K_{K}.txt"))

# with open(w2v_out_path, 'w', encoding='utf-8') as f:
#     for line in lines:
#         f.write(line)


# print("\nTF-IDF ------------------")
# lines = []
# tfidf_metrics = calculate_metrics(expected_results, tfidf_results, K)
# lines.append(f"TF-IDF metrics--> {tfidf_metrics}\n")

# tfidf_average_metrics = calculate_average_model_evaluation_metrics(
#     tfidf_metrics)
# # lines.append(f"TF-IDF average metrics--> {tfidf_average_metrics}\n")

# tfidf_mean_average_metrics = calculate_mean_average_model_evaluation_metrics(
#     tfidf_metrics)
# lines.append(f"TF-IDF mean average metrics--> {tfidf_mean_average_metrics}\n")

# tfidf_out_path = os.path.abspath(
#     os.path.join(out_path, f"tfidf_metrics_K_{K}.txt"))

# with open(tfidf_out_path, 'w', encoding='utf-8') as f:
#     for line in lines:
#         f.write(line)


# print("\nBERT ------------------")
# lines = []
# bert_metrics = calculate_metrics(expected_results, bert_results, K)
# lines.append(f"BERT metrics--> {bert_metrics}\n")

# bert_average_metrics = calculate_average_model_evaluation_metrics(
#     bert_metrics)
# # lines.append(f"BERT average metrics--> {bert_average_metrics}\n")

# bert_mean_average_metrics = calculate_mean_average_model_evaluation_metrics(
#     bert_metrics)
# lines.append(f"BERT mean average metrics--> {bert_mean_average_metrics}\n")

# bert_out_path = os.path.abspath(
#     os.path.join(out_path, f"bert_metrics_K_{K}.txt"))

# with open(bert_out_path, 'w', encoding='utf-8') as f:
#     for line in lines:
#         f.write(line)


# print("\n\nTF-IDF Average metrics: ", tfidf_average_metrics)
# print("\nTF-IDF mean average metrics: ", tfidf_mean_average_metrics)

# print("\n\nw2v Average metrics: ", w2v_average_metrics)
# print("\nw2v mean average metrics: ", w2v_mean_average_metrics)

# print("\n\nBERT Average metrics: ", bert_average_metrics)
# print("\nBERT mean average metrics: ", bert_mean_average_metrics)


# w2v_improvement_metrics = compare_models_evolution_based_on_metrics(
#     tfidf_mean_average_metrics, w2v_mean_average_metrics)

# print("\nWord2Vec improvement metrics against TF-IDF: ", w2v_improvement_metrics)

# bert_improvement_metrics = compare_models_evolution_based_on_metrics(
#     w2v_mean_average_metrics, bert_mean_average_metrics)

# print("\nBERT improvement metrics against W2V: ", bert_improvement_metrics)
