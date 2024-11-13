from pokergame import PokerGame
from ai_model import train_ai, evaluate_ai

def main():
    # Train the AI model
    trained_model = train_ai()

    # Initialize PokerGame with AI player
    player_names = ['User', 'AI']
    game = PokerGame(player_names, trained_model)

    # Evaluate AI's performance
    evaluate_ai(trained_model, game, num_games=100)

    # Start a single game
    game.play_game()

if __name__ == "__main__":
    main()
