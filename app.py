from flask import Flask, request, jsonify
from src.search_engine import SearchEngine

s_engine = SearchEngine()
app = Flask(__name__)


@app.route('/search', methods=['POST'])
def search():
    payload = request.get_json()

    text = payload['text']

    # accepted parameters: embedding_method, similarity_method, rank_count
    embedding_method = request.args['embedding_method']
    similarity_method = request.args['similarity_method']
    rank_count = int(request.args['rank_count'])

    ranking = s_engine.search(text, embedding_method,
                              similarity_method, rank_count)

    response = {
        'ranking': ranking
    }
    print(response)

    return jsonify(response)


app.run(port=5000, debug=True, host='localhost')
