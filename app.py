from flask import Flask, request, jsonify
from src.weight_ranking import get_cos_similarity_weight_ranking, load_tfidf_vectors
from src.embeding_ranking import load_w2v_model, get_cos_similarity_w2v_ranking

app = Flask(__name__)

# pre build dos dados para a busca
anime_tfidf_data = {} # load_tfidf_vectors()
w2v_model = load_w2v_model()

@app.route('/search', methods=['POST'])
def search():
    payload = request.get_json()

    print(request.args.keys())

    text = payload['text']
    ranking_count = int(payload['ranking_count'])

    ranking = get_cos_similarity_weight_ranking(
        text, ranking_count, anime_tfidf_data)

    response = {
        'ranking': ranking
    }

    return jsonify(response)

@app.route('/search2', methods=['POST'])
def search2():
    payload = request.get_json()

    text = payload['text']
    ranking_count = int(payload['ranking_count'])

    ranking = get_cos_similarity_w2v_ranking(
        w2v_model, text, ranking_count)

    response = {
        'ranking': ranking
    }

    return jsonify(response)


app.run(port=5000, debug=True, host='localhost')
