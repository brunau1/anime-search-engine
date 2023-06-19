from src.search_methods.preprocess import read_animes_json, read_animes_json_with_cover
from src.search_methods.tfidf_ranking import TfIdfRanking
from src.search_methods.w2v_ranking import WordToVecRanking


class SearchEngine:
    def __init__(self):
        data_for_processing = read_animes_json()

        self.anime_data = read_animes_json_with_cover()

        self.tfidf_core = TfIdfRanking(
            data_for_processing.get('titles'), data_for_processing.get('sinopses'))
        self.w2v_core = WordToVecRanking(
            data_for_processing.get('titles'), data_for_processing.get('sinopses'))

    def search(self, query, embedding_method='tfidf', similarity_method='cosine', rank_count=10):
        ranking = []
        similar_indexes = []

        titles = self.anime_data.get('names')
        synopses = self.anime_data.get('content')
        cover_urls = self.anime_data.get('coverUrls')

        if embedding_method == 'tfidf':
            similar_indexes = self.tfidf_core.search(
                query, similarity_method, rank_count)

        elif embedding_method == 'w2v':
            similar_indexes = self.w2v_core.search(
                query, similarity_method, rank_count)

        for anime_idx in similar_indexes:
            ranking.append({
                'title': titles[anime_idx],
                'synopsis': synopses[anime_idx],
                'imageUrl': cover_urls[anime_idx]
            })

        return ranking
