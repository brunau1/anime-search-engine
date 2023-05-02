
import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer, models

from services.preprocess import bert_simple_preprocess_text
from services.ranking import cos_similarity_top_results, euclidean_distance_top_results
# from src.search_methods.services.preprocess import bert_simple_preprocess_text
# from src.search_methods.services.ranking import cos_similarity_top_results, euclidean_distance_top_results

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using {DEVICE} DEVICE")


class BertRanking:
    def __init__(self, titles, synopsis, max_len=96, bert_public_path='', device=DEVICE):
        torch.cuda.empty_cache()
        self.max_len = max_len
        self.device = device

        bert_public_path = bert_public_path if bert_public_path != '' else os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', 'public', 'models', 'model_name'))  # add 'temp' for a minor version of bert

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

        self.bert_model.to(self.device)

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
            return_tensors='pt').to(self.device)

        return encoded_synopse

    def get_text_embedding(self, encoded_text):
        with torch.no_grad():
            input_ids = encoded_text["input_ids"].to(self.device)
            mask = encoded_text["attention_mask"].to(self.device)
            params = {'input_ids': input_ids,
                      'attention_mask': mask}

            text_embedding = self.bert_model(params)

        return text_embedding['sentence_embedding'].cpu().numpy()[0]

    def search(self, query, similarity_method="cosine", top_k=10):
        query = bert_simple_preprocess_text(query)

        encoded_query = self.encode_text(query)

        query_embedding = self.get_text_embedding(encoded_query)

        if similarity_method == "cosine":
            ranking = cos_similarity_top_results(
                query_embedding.reshape(1, -1), self.synopsis_embeddings, self.titles, top_k)

            ranking = [[r[0], np.float64(r[1])] for r in ranking]

            return ranking

        elif similarity_method == "euclidean":
            return euclidean_distance_top_results(
                query_embedding, self.synopsis_embeddings, self.titles, top_k)
