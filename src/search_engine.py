from src.preprocess import read_animes_json
from src.tfidf_ranking import TfIdfRanking
from src.w2v_ranking import WordToVecRanking


class SearchEngine:
    def __init__(self):
        anime_data = read_animes_json()

        names = anime_data[0]
        processed_content = anime_data[1]

        self.tfidf_core = TfIdfRanking(names, processed_content)
        self.w2v_core = WordToVecRanking(names, processed_content)

    def search(self, query, embedding_method, similarity_method, rank_count=10):
        if embedding_method == 'tfidf':
            return self.tfidf_core.search(query, similarity_method, rank_count)
        elif embedding_method == 'w2v':
            return self.w2v_core.search(query, similarity_method, rank_count)
