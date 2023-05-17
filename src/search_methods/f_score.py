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

expected_results = [{"id": 0, "relevant_docs": []}, # death note
                    {"id": 1, "relevant_docs": []}, # one punch man
                    {"id": 2, "relevant_docs": []}, # haikyuu
                    {"id": 3, "relevant_docs": []}, # one piece
                    {"id": 4, "relevant_docs": []}, # dragon ball
                    {"id": 5, "relevant_docs": []}, # full metal alchemist
                    {"id": 6, "relevant_docs": []}, # full metal alchemist
                    {"id": 7, "relevant_docs": []}, # naruto
                    {"id": 8, "relevant_docs": []}, # bleach
                    {"id": 9, "relevant_docs": []}] # saint seiya

w2v_results = [{"id": 0, "retrieved_docs": []},
               {"id": 1, "retrieved_docs": []},
               {"id": 2, "retrieved_docs": []},
               {"id": 3, "retrieved_docs": []},
               {"id": 4, "retrieved_docs": []},
               {"id": 5, "retrieved_docs": []},
               {"id": 6, "retrieved_docs": []},
               {"id": 7, "retrieved_docs": []},
               {"id": 8, "retrieved_docs": []},
               {"id": 9, "retrieved_docs": []}]

tfidf_results = [{"id": 0, "retrieved_docs": []},
                 {"id": 1, "retrieved_docs": []},
                 {"id": 2, "retrieved_docs": []},
                 {"id": 3, "retrieved_docs": []},
                 {"id": 4, "retrieved_docs": []},
                 {"id": 5, "retrieved_docs": []},
                 {"id": 6, "retrieved_docs": []},
                 {"id": 7, "retrieved_docs": []},
                 {"id": 8, "retrieved_docs": []},
                 {"id": 9, "retrieved_docs": []}]

bert_results = [{"id": 0, "retrieved_docs": []},
                {"id": 1, "retrieved_docs": []},
                {"id": 2, "retrieved_docs": []},
                {"id": 3, "retrieved_docs": []},
                {"id": 4, "retrieved_docs": []},
                {"id": 5, "retrieved_docs": []},
                {"id": 6, "retrieved_docs": []},
                {"id": 7, "retrieved_docs": []},
                {"id": 8, "retrieved_docs": []},
                {"id": 9, "retrieved_docs": []}]


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


def calculate_f1_score_per_sample(relevant_docs, retrieved_docs):
    return f1_score(relevant_docs, retrieved_docs, average='macro')


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

    for doc in retrieved_docs:
        if doc in relevant_docs:
            pred_labels.append(doc)
        else:
            pred_labels.append(0)

    for doc in relevant_docs:
        true_labels.append(doc)

    # fill true_labels with 0 to pred_labels length
    true_labels += [0] * (len(pred_labels) - len(true_labels))

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
        "f1_score_macro": 0
    }

    for metric in average_metrics:
        print(f"calculating average for {metric}...", np.array(
            model_metrics[metric]).sum())
        average_metrics[metric] += np.array(model_metrics[metric]).sum()
        average_metrics[metric] /= len(model_metrics)

    return average_metrics


def calculate_mean_average_model_evaluation_metrics(model_metrics):
    average_metrics = {
        "precision": 0,
        "recall": 0,
        "f_score": 0,
        "f1_score_binary": 0,
        "f1_score_macro": 0
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
        "f1_score_macro": 0
    }

    for metric in improvement_metrics:
        higger = model1_metrics[metric] if model1_metrics[metric] > model2_metrics[metric] else model2_metrics[metric]
        lower = model1_metrics[metric] if model1_metrics[metric] < model2_metrics[metric] else model2_metrics[metric]

        # calculate percentage of improvement
        improvement_metrics[metric] = (
            abs(higger - lower) / ((higger + lower)/2)) * 100.0

    return improvement_metrics


def calculate_metrics(reference_results, retrieved_results):
    eval_metrics = {
        "precision": [],
        "recall": [],
        "f_score": [],
        "f1_score_binary": [],
        "f1_score_macro": []
    }

    for i, expected_result in enumerate(reference_results):
        print(f"\n\nphrase: {search_phrases[expected_result['id']]}")
        print(f"expected result: {expected_result['relevant_docs']}")
        print(
            f"retrieved result: {retrieved_results[i]['retrieved_docs']}")

        relevant_docs = expected_result["relevant_docs"]
        retrieved_docs = retrieved_results[i]["retrieved_docs"]

        binary_true_labels, binary_pred_labels = get_true_pred_binary_labels(
            relevant_docs, retrieved_docs)

        macro_true_labels, macro_pred_labels = get_true_pred_macro_labels(
            relevant_docs, retrieved_docs)

        eval_metrics["precision"].append(
            calculate_precision(relevant_docs, retrieved_docs))
        eval_metrics["recall"].append(
            calculate_recall(relevant_docs, retrieved_docs))
        eval_metrics["f_score"].append(
            calculate_f_score(relevant_docs, retrieved_docs))
        eval_metrics["f1_score_binary"].append(
            calculate_f1_score_binary(binary_true_labels, binary_pred_labels))
        eval_metrics["f1_score_macro"].append(
            calculate_f1_score_per_sample(macro_true_labels, macro_pred_labels))

        print(f"binary true labels: {binary_true_labels}")
        print(f"binary pred labels: {binary_pred_labels}")
        print(f"macro true labels: {macro_true_labels}")
        print(f"macro pred labels: {macro_pred_labels}")
        print(
            f"precision: {calculate_precision(relevant_docs, retrieved_docs)}")
        print(f"recall: {calculate_recall(relevant_docs, retrieved_docs)}")
        print(f"f-score: {calculate_f_score(relevant_docs, retrieved_docs)}")
        print(
            f"f1-score binary: {calculate_f1_score_binary(binary_true_labels, binary_pred_labels)}")
        print(
            f"f1-score macro: {calculate_f1_score_per_sample(macro_true_labels, macro_pred_labels)}")
        print()

    return eval_metrics


print("\nWord2Vec ------------------")
w2v_metrics = calculate_metrics(expected_results, w2v_results)
w2v_average_metrics = calculate_average_model_evaluation_metrics(w2v_metrics)
w2v_pound_average_metrics = calculate_mean_average_model_evaluation_metrics(
    w2v_metrics)


print("Average metrics: ", w2v_average_metrics)
print("Pound average metrics: ", w2v_pound_average_metrics)

print("\nTF-IDF ------------------")
tfidf_metrics = calculate_metrics(expected_results, tfidf_results)
tfidf_average_metrics = calculate_average_model_evaluation_metrics(
    tfidf_metrics)
tfidf_pound_average_metrics = calculate_mean_average_model_evaluation_metrics(
    tfidf_metrics)

print("Average metrics: ", tfidf_average_metrics)
print("Pound average metrics: ", tfidf_pound_average_metrics)


w2v_improvement_metrics = compare_models_evolution_based_on_metrics(
    tfidf_pound_average_metrics, w2v_pound_average_metrics)

print("\nWord2Vec improvement metrics: ", w2v_improvement_metrics)
