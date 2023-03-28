from flask import Flask, request, jsonify
from src.ranking_tfidf import get_ranking_by_cos_similarity, load_tfidf_vectors

app = Flask(__name__)

# Variável global que armazena os dados pré-processados
anime_tfidf_data = load_tfidf_vectors()


@app.route('/search', methods=['POST'])
def search():
    payload = request.get_json()
    text = payload['text']
    ranking_count = int(payload['ranking_count'])

    ranking = get_ranking_by_cos_similarity(
        text, ranking_count, anime_tfidf_data)

    response = {
        'ranking': ranking
    }

    return jsonify(response)


app.run(port=5000, debug=True, host='localhost')
