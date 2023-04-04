from flask import Flask, request, jsonify
from src.preprocess import read_animes_json
from src.tfidf_ranking import TfIdfRanking
from src.w2v_ranking import WordToVecRanking

app = Flask(__name__)

app.run(port=5000, debug=True, host='localhost')

class App:
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


# @app.route('/search', methods=['POST'])
# def search():
#     payload = request.get_json()

#     print(request.args.keys())

#     text = payload['text']
#     ranking_count = int(payload['ranking_count'])

#     ranking = get_cos_similarity_weight_ranking(
#         text, ranking_count, anime_tfidf_data)

#     response = {
#         'ranking': ranking
#     }

#     return jsonify(response)


# @app.route('/search2', methods=['POST'])
# def search2():
#     payload = request.get_json()

#     text = payload['text']
#     ranking_count = int(payload['ranking_count'])

#     ranking = get_cos_similarity_w2v_ranking(
#         w2v_model, text, ranking_count)

#     response = {
#         'ranking': ranking
#     }

#     return jsonify(response)