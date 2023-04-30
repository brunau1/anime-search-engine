
import os
import re
import torch
import numpy
from tqdm import tqdm
from services.ranking import cos_similarity_top_results, euclidean_distance_top_results
from services.preprocess import read_animes_json, bert_simple_preprocess_text
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer, models
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_SET_SIZE = 5000

torch.cuda.empty_cache()
print(f"Using {DEVICE} DEVICE")


class BertRanking:
    def __init__(self, titles, synopsis):
        self.max_len = 96  # or 128

        bert_public_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', 'public', 'models'))  # add 'temp' for a minor version of bert

        self.tokenizer = BertTokenizer.from_pretrained(
            os.path.join(bert_public_path, 'bert_pretrained_model'), local_files_only=True)

        word_embedding_model = models.Transformer(
            os.path.join(bert_public_path, 'bert_pretrained_model'), max_seq_length=self.max_len)

        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension())

        self.bert_model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model])

        state_dict = torch.load(os.path.join(
            bert_public_path, 'bert_sts_model.pth'))

        self.bert_model.load_state_dict(state_dict, strict=False)

        self.bert_model.eval()

        self.bert_model.to(DEVICE)

        self.similarity_method = torch.nn.CosineSimilarity()

        self.titles = titles

        self.synopsis = [bert_simple_preprocess_text(
            synopse) for synopse in synopsis]

        self.encoded_synopsis = [self.encode_text(
            synopse) for synopse in self.synopsis]

        self.synopsis_embeddings = [self.get_text_embedding(
            encoded_synopse) for _, encoded_synopse in
            tqdm(enumerate(self.encoded_synopsis), desc="Gen embed texts...", position=0)]

    def encode_text(self, text):
        encoded_synopse = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt').to(DEVICE)

        return encoded_synopse

    def get_text_embedding(self, encoded_text):
        with torch.no_grad():
            input_ids = encoded_text["input_ids"].to(DEVICE)
            mask = encoded_text["attention_mask"].to(DEVICE)
            params = {'input_ids': input_ids,
                      'attention_mask': mask}

            text_embedding = self.bert_model(params)

        return text_embedding['sentence_embedding'].cpu().numpy()[0]

    def search(self, query, similarity_method="cosine", top_k=10):
        query = bert_simple_preprocess_text(query)

        encoded_query = self.encode_text(query)

        query_embedding = self.get_text_embedding(encoded_query)

        if similarity_method == "cosine":
            best_similarities = cos_similarity_top_results(
                query_embedding.reshape(1, -1), self.synopsis_embeddings, self.titles, top_k)

        if similarity_method == "euclidean":
            best_similarities = euclidean_distance_top_results(
                query_embedding.reshape(1, -1), self.synopsis_embeddings, self.titles, top_k)

        return best_similarities


public_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'public', 'dataset'))

data = read_animes_json(os.path.join(public_path, 'animes.json'))

titles = data[0][:TRAIN_SET_SIZE]
documents = data[1][:TRAIN_SET_SIZE]

# Define the search phrase for a given anime
search_phrase = "the soldiers fight to protect the goddess athena"
# search_phrase = "the volleyball team is the main focus of the anime"
# search_phrase = "a hero who can defeat any enemy with one punch"
# search_phrase = "it has a dragon wich give three wishes to the one who find it"
# search_phrase = "two brothers enter army to become alchemist"
# search_phrase = "a ninja boy who wants to become the leader of his village"
# search_phrase = "give me an anime about giant robots"
# search_phrase = "the protagonist got the shinigami sword and now he has to kill hollows"
# search_phrase = "the protagonist dies and reincarnates as a slime"

rank_model = BertRanking(titles, documents)

print(rank_model.search(search_phrase, "cosine", 10))
