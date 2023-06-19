from flask import Flask, request, jsonify
from flask_cors import CORS
from src.search_engine import SearchEngine

s_engine = SearchEngine()
app = Flask(__name__)
CORS(app)

@app.route('/search', methods=['GET'])
def search():
    similarity_method = 'cosine'

    text = request.args['text']
    embedding_method = request.args['embedding']
    rank_count = int(request.args['limit'])

    ranking = s_engine.search(text, embedding_method,
                              similarity_method, rank_count)

    response = {
        'ranking': ranking
    }

    print("returning response. len(ranking) = " + str(len(ranking)) + " rank_count = " + str(rank_count))

    return jsonify(response)


app.run(port=5000, debug=True, host='localhost')
