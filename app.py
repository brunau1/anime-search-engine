from flask import Flask, request, jsonify
from src.search_engine import SearchEngine

s_engine = SearchEngine()
app = Flask(__name__)


@app.route('/search', methods=['POST'])
def search():
    payload = request.get_json()

    # accepted parameters: embedding_method, similarity_method, rank_count
    print(request.args.keys())

    text = payload['text']
    # ranking_count = int(payload['ranking_count'])

    ranking = s_engine.search(text, 'tfidf', 'cosine')

    response = {
        'ranking': ranking
    }

    return jsonify(response)


app.run(port=5000, debug=True, host='localhost')
