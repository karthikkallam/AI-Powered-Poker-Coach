from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

@app.route('/start_game', methods=['GET'])
def start_game():
    initial_state = {
        'pot': 0,
        'player_stack': 100,
        'ai_stack': 100,
        'community_cards': [],
        'player_cards': []
    }
    return jsonify(initial_state)

@app.route('/bet', methods=['POST'])
def bet():
    amount = request.json.get('amount', 0)
    # Logic to handle bet action
    # Return updated game state
    return jsonify({"message": f"Bet {amount} chips"})

@app.route('/call', methods=['POST'])
def call():
    # Logic to handle call action
    return jsonify({"message": "Called the bet"})

@app.route('/raise', methods=['POST'])
def raise_bet():
    amount = request.json.get('amount', 0)
    # Logic to handle raise action
    return jsonify({"message": f"Raised by {amount} chips"})

@app.route('/fold', methods=['POST'])
def fold():
    # Logic to handle fold action
    return jsonify({"message": "Folded"})
