import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os


class TicTacToeNet(nn.Module):
    def __init__(self):
        super(TicTacToeNet, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 9)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class TicTacToeAI:
    def __init__(self, model_path='models/tictactoe_model.pth'):
        self.model = TicTacToeNet()
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def load_and_prepare_data(self, csv_path):
        """Load and prepare the Tic-Tac-Toe dataset"""
        df = pd.read_csv(csv_path)
        
        # Create board states from moves
        X = []
        y = []
        
        for _, row in df.iterrows():
            board = np.zeros(9, dtype=int)  # 0: empty, 1: X (player), -1: O (AI)
            moves = []
            
            # Extract moves
            for i in range(1, 8):
                col = f'MOVE{i}'
                if col in row and pd.notna(row[col]) and row[col] != '?':
                    moves.append(int(row[col]))
            
            # Build the game state and predict next move
            for move_idx in range(len(moves)):
                if move_idx > 0:  # We need at least one move to make a prediction
                    # Apply moves alternately (X and O)
                    for j in range(move_idx):
                        pos = moves[j]
                        board[pos] = 1 if j % 2 == 0 else -1
                    
                    # Next move to predict
                    if move_idx < len(moves):
                        next_move = moves[move_idx]
                        X.append(board.copy())
                        y.append(next_move)
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)
    
    def train(self, csv_path, epochs=100, batch_size=32, lr=0.001):
        """Train the Tic-Tac-Toe AI model"""
        print(f"Loading data from {csv_path}...")
        X, y = self.load_and_prepare_data(csv_path)
        
        print(f"Dataset size: {len(X)} samples")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.LongTensor(y_val).to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        print("Training model...")
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            
            # Mini-batch training
            total_loss = 0
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val)
                
                # Calculate accuracy
                _, predicted = torch.max(val_outputs, 1)
                accuracy = (predicted == y_val).sum().item() / len(y_val)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], "
                      f"Loss: {total_loss/len(X_train):.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Accuracy: {accuracy:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model()
        
        print("Training completed!")
        return accuracy
    
    def predict_move(self, board_state):
        """
        Predict the best move for the AI
        board_state: list or array of 9 elements (0: empty, 1: X, -1: O)
        Returns: position (0-8) for the next move
        """
        self.model.eval()
        
        # Convert board to tensor
        board_tensor = torch.FloatTensor(board_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(board_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
        
        # Get valid moves (empty positions)
        valid_moves = [i for i, val in enumerate(board_state) if val == 0]
        
        if not valid_moves:
            return None
        
        # Sort moves by probability
        move_probs = [(i, probabilities[i].item()) for i in valid_moves]
        move_probs.sort(key=lambda x: x[1], reverse=True)
        
        return move_probs[0][0]
    
    def save_model(self):
        """Save the trained model"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load a trained model"""
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"Model loaded from {self.model_path}")
            return True
        return False


def check_winner(board):
    """Check if there's a winner"""
    # Winning combinations
    wins = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
        [0, 4, 8], [2, 4, 6]              # diagonals
    ]
    
    for combo in wins:
        if board[combo[0]] == board[combo[1]] == board[combo[2]] != 0:
            return board[combo[0]]
    
    if 0 not in board:
        return 0  # Draw
    
    return None  # Game continues


def print_board(board):
    """Print the board in a readable format"""
    symbols = {0: '.', 1: 'X', -1: 'O'}
    print("\n")
    for i in range(0, 9, 3):
        print(f" {symbols[board[i]]} | {symbols[board[i+1]]} | {symbols[board[i+2]]} ")
        if i < 6:
            print("-----------")
    print("\n")


if __name__ == "__main__":
    # Train the model
    ai = TicTacToeAI()
    dataset_path = "/Users/kmj5004/Downloads/Tic tac initial results.csv"
    
    if os.path.exists(dataset_path):
        accuracy = ai.train(dataset_path, epochs=100)
        print(f"\nFinal model accuracy: {accuracy:.4f}")
        
        # Test the AI
        print("\n=== Testing AI ===")
        ai.load_model()
        
        # Example game
        board = [0] * 9
        print("Starting a test game (X is human, O is AI):")
        print_board(board)
        
        # Simulate a few moves
        test_moves = [4, 0, 1]  # Human plays center, AI responds, Human plays top-left
        for i, move in enumerate(test_moves):
            if i % 2 == 0:
                board[move] = 1
                print(f"Human plays position {move}")
            else:
                board[move] = -1
                print(f"AI plays position {move}")
            print_board(board)
            
            winner = check_winner(board)
            if winner is not None:
                break
            
            if i % 2 == 0:
                ai_move = ai.predict_move(board)
                if ai_move is not None:
                    board[ai_move] = -1
                    print(f"AI predicts move: {ai_move}")
                    print_board(board)
    else:
        print(f"Dataset not found at {dataset_path}")
