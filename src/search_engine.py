from src.preprocess import read_animes_json
from src.tfidf_ranking import TfIdfRanking
from src.w2v_ranking import WordToVecRanking


class SearchEngine:
    def __init__(self):
        anime_data = read_animes_json()

        titles = anime_data[0]
        sinopsis = anime_data[1]

        self.tfidf_core = TfIdfRanking(titles, sinopsis)
        self.w2v_core = WordToVecRanking(titles, sinopsis)

    def search(self, query, embedding_method='tfidf', similarity_method='cosine', rank_count=10):
        if embedding_method == 'tfidf':
            return self.tfidf_core.search(query, similarity_method, rank_count)
        elif embedding_method == 'w2v':
            return self.w2v_core.search(query, similarity_method, rank_count)
