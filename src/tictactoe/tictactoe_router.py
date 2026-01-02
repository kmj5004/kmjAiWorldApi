from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from src.tictactoe.tictactoe_ai import TicTacToeAI, check_winner
import os

router = APIRouter(prefix="/tictactoe", tags=["TicTacToe AI"])

# Initialize AI model
ai_model = TicTacToeAI(model_path='models/tictactoe_model.pth')
model_loaded = ai_model.load_model()


class BoardState(BaseModel):
    board: List[int] = Field(..., description="Board state as 9 integers: 0=empty, 1=X (human), -1=O (AI)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "board": [1, 0, 0, 0, -1, 0, 0, 0, 0]
            }
        }


class MoveResponse(BaseModel):
    move: Optional[int] = Field(None, description="AI's chosen move position (0-8)")
    board: List[int] = Field(..., description="Updated board state")
    winner: Optional[int] = Field(None, description="Winner if game ended: 1=X wins, -1=O wins, 0=draw, None=continues")
    message: str = Field(..., description="Response message")


class GameState(BaseModel):
    board: List[int]
    status: str
    winner: Optional[int]
    next_player: str


@router.post("/move", response_model=MoveResponse)
async def ai_move(board_state: BoardState):
    """
    Get AI's next move for the given board state
    
    - **board**: Current board state (9 integers)
    - Returns AI's move and updated board state
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="AI model not loaded. Please train the model first.")
    
    board = board_state.board
    
    # Validate board
    if len(board) != 9:
        raise HTTPException(status_code=400, detail="Board must have exactly 9 positions")
    
    if not all(x in [0, 1, -1] for x in board):
        raise HTTPException(status_code=400, detail="Board values must be 0 (empty), 1 (X), or -1 (O)")
    
    # Check if game already ended
    winner = check_winner(board)
    if winner is not None:
        return MoveResponse(
            move=None,
            board=board,
            winner=winner,
            message="Game already ended"
        )
    
    # Get AI move
    try:
        ai_move = ai_model.predict_move(board)
        
        if ai_move is None:
            return MoveResponse(
                move=None,
                board=board,
                winner=0,
                message="No valid moves available (Draw)"
            )
        
        # Apply AI move
        new_board = board.copy()
        new_board[ai_move] = -1
        
        # Check for winner after AI move
        winner = check_winner(new_board)
        
        message = f"AI plays position {ai_move}"
        if winner == -1:
            message += " - AI wins!"
        elif winner == 0:
            message += " - Game is a draw!"
        
        return MoveResponse(
            move=ai_move,
            board=new_board,
            winner=winner,
            message=message
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting move: {str(e)}")


@router.get("/new-game", response_model=GameState)
async def new_game():
    """Start a new Tic-Tac-Toe game"""
    return GameState(
        board=[0] * 9,
        status="active",
        winner=None,
        next_player="human"
    )


@router.post("/check-winner")
async def check_game_status(board_state: BoardState):
    """Check the current game status"""
    board = board_state.board
    
    if len(board) != 9:
        raise HTTPException(status_code=400, detail="Board must have exactly 9 positions")
    
    winner = check_winner(board)
    
    if winner == 1:
        status = "X wins!"
    elif winner == -1:
        status = "O wins!"
    elif winner == 0:
        status = "Draw"
    else:
        status = "Game in progress"
    
    return {
        "board": board,
        "winner": winner,
        "status": status
    }


@router.get("/train-status")
async def get_train_status():
    """Get model training status"""
    model_path = 'models/tictactoe_model.pth'
    exists = os.path.exists(model_path)
    
    return {
        "model_exists": exists,
        "model_loaded": model_loaded,
        "model_path": model_path,
        "status": "ready" if model_loaded else "not trained"
    }


@router.get("/board-representation")
async def get_board_representation():
    """Get information about board representation"""
    return {
        "board_size": 9,
        "positions": {
            "layout": [
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8]
            ],
            "description": "Position indices from 0-8"
        },
        "values": {
            "0": "Empty",
            "1": "X (Human player)",
            "-1": "O (AI player)"
        },
        "example": {
            "board": [1, 0, 0, 0, -1, 0, 0, 0, 0],
            "visualization": [
                ["X", ".", "."],
                [".", "O", "."],
                [".", ".", "."]
            ]
        }
    }
