import os
import numpy as np
from sklearn.metrics import f1_score

from preprocess import preprocess_text
'''
- determinar os conjuntos com os dez indices de texto esperados para cada frase de entrada
- instanciar o modelo de similaridade e calcular as similaridades
- a partir dos indices dos textos mais similares, criar uma lista de valores
binarios em que 1 indica que o texto foi recuperado e 0 indica que o texto nao foi recuperado
- calcular a precisao, revocacao e f-score para cada frase de entrada
'''

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

expected_results = [{"id": 0, "relevant_docs": [726]},  # death note
                    # one punch man
                    {"id": 1, "relevant_docs": [
                        16, 863, 8583, 10096, 8939, 881]},
                    # haikyuu e attack 1
                    {"id": 2, "relevant_docs": [721, 0, 6727, 13695, 10076]},
                    {"id": 3, "relevant_docs": [
                        10098, 14996, 312, 519, 471, 3859, 932, 238, 321, 5932]},  # one piece
                    {"id": 4, "relevant_docs": [
                        370, 819, 7420, 9999, 6024, 5432, 2827, 1273, 8851, 14222]},  # dragon ball
                    # full metal alchemist
                    {"id": 5, "relevant_docs": [
                        499, 3, 340, 13378, 2894, 5886, 12908, 5531, 14660]},
                    # full metal alchemist
                    {"id": 6, "relevant_docs": [
                        499, 3, 340, 13378, 2894, 5886, 12908, 5531, 14660]},
                    {"id": 7, "relevant_docs": [
                        142, 473, 3842, 4786, 13468, 802, 906, 3644, 4499, 5144]},  # naruto
                    {"id": 8, "relevant_docs": [
                        57, 15084, 4998, 9469, 6560, 5478, 3103, 10161]},  # bleach
                    {"id": 9, "relevant_docs": [7208, 111, 11987, 14598, 394, 21, 2940, 348, 3165, 15152]}]  # saint seiya

w2v_results = [{"id": 0, "retrieved_docs":
                [726, 1973, 11047, 1024, 1086, 12737, 14390, 354, 11748, 1096]},
               {"id": 1, "retrieved_docs": [
                   16, 10770, 863, 10180, 1425, 9403, 14666, 720, 5681, 10676]},
               {"id": 2, "retrieved_docs": [
                   13695, 10076, 9009, 3042, 6727, 3293, 2462, 0, 721, 11635]},
               {"id": 3, "retrieved_docs": [
                   10098, 5932, 14783, 3859, 3421, 471, 237, 14672, 238, 503]},
               {"id": 4, "retrieved_docs": [
                   6615, 6792, 11150, 13193, 2726, 11872, 4372, 14222, 9138, 370]},
               {"id": 5, "retrieved_docs": [
                   13378, 499, 3, 14636, 14690, 155, 2088, 6615, 12922, 655]},
               {"id": 6, "retrieved_docs": [
                   13378, 499, 3, 13154, 12922, 8561, 155, 2088, 14025, 6615]},
               {"id": 7, "retrieved_docs": [
                   14006, 9666, 11180, 1559, 5594, 3921, 12079, 12491, 6526, 5164]},
               {"id": 8, "retrieved_docs": [
                   57, 5344, 15084, 4998, 3103, 202, 8591, 9469, 10826, 5478]},
               {"id": 9, "retrieved_docs": [111, 11987, 14598, 15152, 7208, 21, 394, 348, 4920, 2940]}]

tfidf_results = [{"id": 0, "retrieved_docs": [1973, 726, 1550, 2140, 934, 9056, 3163, 7448, 1675, 1086]},
                 {"id": 1, "retrieved_docs": [
                     863, 10096, 8939, 16, 14666, 10136, 10770, 9403, 6615, 1425]},
                 {"id": 2, "retrieved_docs": [
                     9009, 10076, 13695, 6727, 0, 721, 13532, 2462, 2519, 3042]},
                 {"id": 3, "retrieved_docs": [
                     10098, 5932, 10363, 14783, 471, 15100, 519, 3859, 14999, 3421]},
                 {"id": 4, "retrieved_docs": [
                     14222, 9196, 9006, 2726, 6310, 11872, 370, 8296, 8903, 9488]},
                 {"id": 5, "retrieved_docs": [
                     12922, 13154, 14025, 2088, 11550, 3894, 5919, 10196, 5280, 8561]},
                 {"id": 6, "retrieved_docs": [
                     12922, 2088, 3, 499, 11550, 13154, 14025, 13378, 9418, 8561]},
                 {"id": 7, "retrieved_docs": [
                     9666, 1559, 11182, 14006, 13128, 5113, 11807, 12491, 12079, 10971]},
                 {"id": 8, "retrieved_docs": [
                     8591, 10826, 57, 15084, 1550, 5344, 2140, 934, 726, 7727]},
                 {"id": 9, "retrieved_docs": [14598, 111, 11987, 7208, 348, 15152, 394, 12468, 5180, 14529]}]

bert_results = [{"id": 0, "retrieved_docs": [726, 3163, 3130, 3106, 12375, 1024, 5507, 10988, 11815, 5466]},
                {"id": 1, "retrieved_docs": [
                    16, 720, 863, 5905, 240, 8939, 428, 1076, 11402, 10096]},
                {"id": 2, "retrieved_docs": [
                    3042, 13695, 721, 6727, 0, 11138, 10464, 10076, 9621, 5211]},
                {"id": 3, "retrieved_docs": [
                    471, 10098, 3859, 932, 15100, 5932, 312, 14996, 238, 321]},
                {"id": 4, "retrieved_docs": [
                    7420, 9999, 2827, 1273, 8906, 819, 6024, 5432, 3316, 8851]},
                {"id": 5, "retrieved_docs": [
                    499, 13154, 3, 2894, 6882, 8852, 10298, 9523, 5037, 11655]},
                {"id": 6, "retrieved_docs": [
                    3, 2894, 12908, 340, 499, 1630, 5886, 11834, 13378, 12330]},
                {"id": 7, "retrieved_docs": [
                    3564, 142, 10962, 3842, 11180, 6382, 4786, 12491, 10066, 13468]},
                {"id": 8, "retrieved_docs": [
                    15084, 57, 4998, 9469, 5478, 502, 3103, 7657, 9382, 7920]},
                {"id": 9, "retrieved_docs": [7208, 348, 21, 14598, 394, 3165, 15152, 781, 2940, 14315]}]


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


def calculate_metrics(reference_results, retrieved_results, K=10):
    eval_metrics = {
        "precision": [],
        "recall": [],
        "f_score": [],
        "f1_score_binary": [],
        "f1_score_macro": [],
        "f1_score_micro": [],
        "f1_score_weighted": []
    }

    for i, expected_result in enumerate(reference_results):
        print(f"\n\nphrase: {search_phrases[expected_result['id']]}")
        print(f"expected result: {expected_result['relevant_docs']}")
        print(
            f"retrieved result: {retrieved_results[i]['retrieved_docs']}")

        relevant_docs = expected_result["relevant_docs"]
        retrieved_docs = retrieved_results[i]["retrieved_docs"][:K]

        # pick each doc in relevant_docs and check if it is in retrieved_docs and append the ids in an separate array
        curr_relevant_docs = []
        for doc in relevant_docs:
            if doc in retrieved_docs:
                curr_relevant_docs.append(doc)

        # the relevant docs is the current relevant docs plus the rest of the relevant docs that are not in current relevant docs
        relevant_docs = curr_relevant_docs + \
            list(set(relevant_docs) - set(curr_relevant_docs))

        # now pick the K first docs in relevant_docs
        relevant_docs = relevant_docs[:K]

        binary_true_labels, binary_pred_labels = get_true_pred_binary_labels(
            relevant_docs, retrieved_docs)

        macro_true_labels, macro_pred_labels = get_true_pred_macro_labels(
            relevant_docs, retrieved_docs)

        print(f"binary true labels: {binary_true_labels}")
        print(f"binary pred labels: {binary_pred_labels}")
        print(f"macro true labels: {macro_true_labels}")
        print(f"macro pred labels: {macro_pred_labels}")

        eval_metrics["precision"].append(
            calculate_precision(relevant_docs, retrieved_docs))
        eval_metrics["recall"].append(
            calculate_recall(relevant_docs, retrieved_docs))
        eval_metrics["f_score"].append(
            calculate_f_score(relevant_docs, retrieved_docs))
        eval_metrics["f1_score_binary"].append(
            calculate_f1_score_binary(binary_true_labels, binary_pred_labels))
        eval_metrics["f1_score_macro"].append(
            calculate_f1_score_macro(macro_true_labels, macro_pred_labels))
        eval_metrics["f1_score_micro"].append(
            calculate_f1_score_micro(macro_true_labels, macro_pred_labels))
        eval_metrics["f1_score_weighted"].append(
            calculate_f1_score_weighted(macro_true_labels, macro_pred_labels))

        print(
            f"precision: {calculate_precision(relevant_docs, retrieved_docs)}")
        print(f"recall: {calculate_recall(relevant_docs, retrieved_docs)}")
        print(f"f-score: {calculate_f_score(relevant_docs, retrieved_docs)}")
        print(
            f"f1-score binary: {calculate_f1_score_binary(binary_true_labels, binary_pred_labels)}")
        print(
            f"f1-score macro: {calculate_f1_score_macro(macro_true_labels, macro_pred_labels)}")
        print(
            f"f1-score micro: {calculate_f1_score_micro(macro_true_labels, macro_pred_labels)}")
        print(
            f"f1-score weighted: {calculate_f1_score_weighted(macro_true_labels, macro_pred_labels)}")
        print()

    return eval_metrics


out_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'public', 'dataset', 'evaluation'))

K = 10
lines = []

print("\nWord2Vec ------------------")
w2v_metrics = calculate_metrics(expected_results, w2v_results, K)
lines.append(f"Word2Vec metrics--> {w2v_metrics}\n")

w2v_average_metrics = calculate_average_model_evaluation_metrics(w2v_metrics)
# lines.append(f"Word2Vec average metrics--> {w2v_average_metrics}\n")

w2v_mean_average_metrics = calculate_mean_average_model_evaluation_metrics(
    w2v_metrics)
lines.append(f"Word2Vec mean average metrics--> {w2v_mean_average_metrics}\n")

w2v_out_path = os.path.abspath(
    os.path.join(out_path, f"w2v_metrics_K_{K}.txt"))

with open(w2v_out_path, 'w', encoding='utf-8') as f:
    for line in lines:
        f.write(line)


print("\nTF-IDF ------------------")
lines = []
tfidf_metrics = calculate_metrics(expected_results, tfidf_results, K)
lines.append(f"TF-IDF metrics--> {tfidf_metrics}\n")

tfidf_average_metrics = calculate_average_model_evaluation_metrics(
    tfidf_metrics)
# lines.append(f"TF-IDF average metrics--> {tfidf_average_metrics}\n")

tfidf_mean_average_metrics = calculate_mean_average_model_evaluation_metrics(
    tfidf_metrics)
lines.append(f"TF-IDF mean average metrics--> {tfidf_mean_average_metrics}\n")

tfidf_out_path = os.path.abspath(
    os.path.join(out_path, f"tfidf_metrics_K_{K}.txt"))

with open(tfidf_out_path, 'w', encoding='utf-8') as f:
    for line in lines:
        f.write(line)


print("\nBERT ------------------")
lines = []
bert_metrics = calculate_metrics(expected_results, bert_results, K)
lines.append(f"BERT metrics--> {bert_metrics}\n")

bert_average_metrics = calculate_average_model_evaluation_metrics(
    bert_metrics)
# lines.append(f"BERT average metrics--> {bert_average_metrics}\n")

bert_mean_average_metrics = calculate_mean_average_model_evaluation_metrics(
    bert_metrics)
lines.append(f"BERT mean average metrics--> {bert_mean_average_metrics}\n")

bert_out_path = os.path.abspath(
    os.path.join(out_path, f"bert_metrics_K_{K}.txt"))

with open(bert_out_path, 'w', encoding='utf-8') as f:
    for line in lines:
        f.write(line)


print("\n\nTF-IDF Average metrics: ", tfidf_average_metrics)
print("\nTF-IDF mean average metrics: ", tfidf_mean_average_metrics)

print("\n\nw2v Average metrics: ", w2v_average_metrics)
print("\nw2v mean average metrics: ", w2v_mean_average_metrics)

print("\n\nBERT Average metrics: ", bert_average_metrics)
print("\nBERT mean average metrics: ", bert_mean_average_metrics)


w2v_improvement_metrics = compare_models_evolution_based_on_metrics(
    tfidf_mean_average_metrics, w2v_mean_average_metrics)

print("\nWord2Vec improvement metrics against TF-IDF: ", w2v_improvement_metrics)

bert_improvement_metrics = compare_models_evolution_based_on_metrics(
    w2v_mean_average_metrics, bert_mean_average_metrics)

print("\nBERT improvement metrics against W2V: ", bert_improvement_metrics)
