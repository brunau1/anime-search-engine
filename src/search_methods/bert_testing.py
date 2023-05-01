import os
from bert_ranking import BertRanking

from services.preprocess import read_animes_json

def calc_and_save_results(model, file_name):
    method = "cosine"

    search_phrases = ["the soldiers fight to protect the goddess athena",
                      "the protagonist is a demon who wants to become a hero",
                      "the volleyball team is the main focus of the anime",
                      "a man who can defeat any enemy with one punch",
                      "it has a dragon wich give three wishes to the one who find it",
                      "two brothers enter army to become alchemist",
                      "a ninja boy who wants to become a hokage",
                      "the protagonist got the shinigami sword and now he has to kill hollows",
                      "give me an anime about giant robots"]

    lines = []
    for search_phrase in search_phrases:
        line = model.search(search_phrase, method, 10)
        lines.append(f"''{search_phrase}'' -> Results: {line}")

    with open(file_name, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(f"{line}\n")


public_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'public', 'dataset'))

data = read_animes_json(os.path.join(public_path, 'animes.json'))

titles = data[0]  # [:TRAIN_SET_SIZE]
documents = data[1]  # [:TRAIN_SET_SIZE]

bert_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'public', 'models'))

dataset_path = os.path.abspath(os.path.join(public_path, 'search_results'))

TRAIN_SET_SIZE = 1000

print("Loading model with", TRAIN_SET_SIZE, "animes...")

bert_64_1000 = BertRanking(
    titles[:TRAIN_SET_SIZE], documents[:TRAIN_SET_SIZE], os.path.join(bert_path, 'bert_64_1000'))

calc_and_save_results(bert_64_1000, os.path.join(
    dataset_path, 'bert_64_1000.txt'))


TRAIN_SET_SIZE = 5000

print("Loading model with", TRAIN_SET_SIZE, "animes...")

bert_64_5000 = BertRanking(
    titles[:TRAIN_SET_SIZE], documents[:TRAIN_SET_SIZE], os.path.join(bert_path, 'bert_64_5000'))

calc_and_save_results(bert_64_5000, os.path.join(
    dataset_path, 'bert_64_5000.txt'))


TRAIN_SET_SIZE = 10000

print("Loading model with", TRAIN_SET_SIZE, "animes...")

bert_64_10000 = BertRanking(
    titles[:TRAIN_SET_SIZE], documents[:TRAIN_SET_SIZE], os.path.join(bert_path, 'bert_64_10000'))

calc_and_save_results(bert_64_10000, os.path.join(
    dataset_path, 'bert_64_10000.txt'))


TRAIN_SET_SIZE = "all"

print("Loading model with", TRAIN_SET_SIZE, "animes...")

bert_64_15000 = BertRanking(
    titles, documents, os.path.join(bert_path, 'bert_64_15000'))

calc_and_save_results(bert_64_15000, os.path.join(
    dataset_path, 'bert_64_15000.txt'))


TRAIN_SET_SIZE = 1000

print("Loading model with", TRAIN_SET_SIZE, "animes...")

bert_96_1000 = BertRanking(
    titles[:TRAIN_SET_SIZE], documents[:TRAIN_SET_SIZE], os.path.join(bert_path, 'bert_96_1000'))

calc_and_save_results(bert_96_1000, os.path.join(
    dataset_path, 'bert_96_1000.txt'))


TRAIN_SET_SIZE = 5000

print("Loading model with", TRAIN_SET_SIZE, "animes...")

bert_96_5000 = BertRanking(
    titles[:TRAIN_SET_SIZE], documents[:TRAIN_SET_SIZE], os.path.join(bert_path, 'bert_96_5000'))

calc_and_save_results(bert_96_5000, os.path.join(
    dataset_path, 'bert_96_5000.txt'))


TRAIN_SET_SIZE = 10000

print("Loading model with", TRAIN_SET_SIZE, "animes...")

bert_96_10000 = BertRanking(
    titles[:TRAIN_SET_SIZE], documents[:TRAIN_SET_SIZE], os.path.join(bert_path, 'bert_96_10000'))

calc_and_save_results(bert_96_10000, os.path.join(
    dataset_path, 'bert_96_10000.txt'))


TRAIN_SET_SIZE = "all"

print("Loading model with", TRAIN_SET_SIZE, "animes...")

bert_96_15000 = BertRanking(
    titles, documents, os.path.join(bert_path, 'bert_96_15000'))

calc_and_save_results(bert_96_15000, os.path.join(
    dataset_path, 'bert_96_15000.txt'))


TRAIN_SET_SIZE = 1000

print("Loading model with", TRAIN_SET_SIZE, "animes...")

bert_128_1000 = BertRanking(
    titles[:TRAIN_SET_SIZE], documents[:TRAIN_SET_SIZE], os.path.join(bert_path, 'bert_128_1000'))

calc_and_save_results(bert_128_1000, os.path.join(
    dataset_path, 'bert_128_1000.txt'))


TRAIN_SET_SIZE = 5000

print("Loading model with", TRAIN_SET_SIZE, "animes...")

bert_128_5000 = BertRanking(
    titles[:TRAIN_SET_SIZE], documents[:TRAIN_SET_SIZE], os.path.join(bert_path, 'bert_128_5000'))

calc_and_save_results(bert_128_5000, os.path.join(
    dataset_path, 'bert_128_5000.txt'))


TRAIN_SET_SIZE = 10000

print("Loading model with", TRAIN_SET_SIZE, "animes...")

bert_128_10000 = BertRanking(
    titles[:TRAIN_SET_SIZE], documents[:TRAIN_SET_SIZE], os.path.join(bert_path, 'bert_128_10000'))

calc_and_save_results(bert_128_10000, os.path.join(
    dataset_path, 'bert_128_10000.txt'))


TRAIN_SET_SIZE = "all"

print("Loading model with", TRAIN_SET_SIZE, "animes...")

bert_128_15000 = BertRanking(
    titles, documents, os.path.join(bert_path, 'bert_128_15000'))

calc_and_save_results(bert_128_15000, os.path.join(
    dataset_path, 'bert_128_15000.txt'))

