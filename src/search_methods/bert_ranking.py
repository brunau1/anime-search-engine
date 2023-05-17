
import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer, models

from preprocess import bert_simple_preprocess_text, simple_preprocess_text
from ranking import cos_similarity_top_results, euclidean_distance_top_results
# from src.search_methods.services.preprocess import bert_simple_preprocess_text
# from src.search_methods.services.ranking import cos_similarity_top_results, euclidean_distance_top_results

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using {DEVICE} DEVICE")
torch.cuda.empty_cache()


class BertRanking:
    def __init__(self, titles, synopsis, max_len=128, bert_public_path='', device=DEVICE):
        self.max_len = max_len
        self.device = device

        self.public_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', 'public', 'dataset'))

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

        loaded_embeddings = []
        # loaded_embeddings = np.load(os.path.join(public_path, 'modules', 'bert_trained_all_database',
        #                                          'bert_anime_embeddings_all.npy'))

        self.synopsis = [bert_simple_preprocess_text(
            synopse) for synopse in synopsis] if len(loaded_embeddings) == 0 else []

        self.encoded_synopsis = [self.encode_text(synopse) for _, synopse in tqdm(
            enumerate(self.synopsis), position=0)] if len(loaded_embeddings) == 0 else []

        self.synopsis_embeddings = loaded_embeddings if len(
            loaded_embeddings) > 0 else self.get_texts_embeddings()

    def encode_text(self, text):
        encoded_synopse = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt')

        return encoded_synopse

    def get_text_embedding(self, encoded_text):
        with torch.no_grad():
            input_ids = encoded_text["input_ids"].to(self.device)
            mask = encoded_text["attention_mask"].to(self.device)
            params = {'input_ids': input_ids,
                      'attention_mask': mask}

            text_embedding = self.bert_model(params)

        return text_embedding['sentence_embedding'].cpu().numpy()[0]

    def get_texts_embeddings(self):
        texts_embeddings = []
        with torch.no_grad():
            for _, encoded_text in tqdm(enumerate(self.encoded_synopsis), position=0):
                input_ids = encoded_text["input_ids"].to(self.device)
                mask = encoded_text["attention_mask"].to(self.device)
                params = {'input_ids': input_ids,
                          'attention_mask': mask}

                text_embedding = self.bert_model(params)

                texts_embeddings.append(
                    text_embedding['sentence_embedding'].cpu().numpy()[0])

        text_embeddings = np.array(texts_embeddings)
        np.save(os.path.join(self.public_path, 'modules', 'bert_trained_all_database',
                             'bert_anime_embeddings_all.npy'), texts_embeddings)

        return text_embeddings

    def search(self, query, similarity_method="cosine", top_k=10):
        query = bert_simple_preprocess_text(query)

        encoded_query = self.encode_text("[CLS] " + " ".join(query) + " [SEP]")

        query_embedding = self.get_text_embedding(encoded_query)

        if similarity_method == "cosine":
            ranking = cos_similarity_top_results(
                query_embedding.reshape(1, -1), self.synopsis_embeddings, self.titles, top_k)

            ranking = [[r[0], np.float64(r[1])] for r in ranking]

            return ranking

        elif similarity_method == "euclidean":
            return euclidean_distance_top_results(
                query_embedding, self.synopsis_embeddings, self.titles, top_k)
