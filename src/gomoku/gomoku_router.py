"""
오목 AI API 라우터
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import torch
import os

from src.gomoku.game import GomokuGame
from src.gomoku.network import GomokuNet
from src.gomoku.mcts import MCTS

router = APIRouter(prefix="/gomoku", tags=["Gomoku AI"])

# AI 모델 초기화
BOARD_SIZE = 15
MODEL_PATH = 'models/gomoku/gomoku_net.pth'

network = GomokuNet(board_size=BOARD_SIZE, num_channels=128, num_res_blocks=5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network.to(device)

# 모델 로드 시도
model_loaded = False
if os.path.exists(MODEL_PATH):
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        network.load_state_dict(checkpoint['model_state_dict'])
        network.eval()
        model_loaded = True
        print(f"Gomoku model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"Warning: Could not load Gomoku model: {e}")
else:
    print("Gomoku model not found. AI will use untrained network.")

mcts = MCTS(network, num_simulations=400)

# 게임 세션 관리 (간단한 인메모리 저장소)
game_sessions = {}

class BoardState(BaseModel):
    board: List[List[int]] = Field(..., description="15x15 board: 0=empty, 1=black, -1=white")
    current_player: int = Field(..., description="Current player: 1=black, -1=white")

class MoveRequest(BaseModel):
    game_id: Optional[str] = None
    board: Optional[List[List[int]]] = None
    current_player: Optional[int] = None
    difficulty: str = Field(default="normal", description="easy/normal/hard")

class MoveResponse(BaseModel):
    game_id: str
    row: int
    col: int
    board: List[List[int]]
    winner: Optional[int]
    game_over: bool
    message: str

class NewGameResponse(BaseModel):
    game_id: str
    board: List[List[int]]
    board_size: int
    current_player: int
    message: str

@router.get("/new-game", response_model=NewGameResponse)
async def new_game():
    """새 오목 게임 시작"""
    import uuid
    game_id = str(uuid.uuid4())
    
    game = GomokuGame(board_size=BOARD_SIZE)
    game.reset()
    game_sessions[game_id] = game
    
    return NewGameResponse(
        game_id=game_id,
        board=game.board.tolist(),
        board_size=BOARD_SIZE,
        current_player=game.current_player,
        message="New game started"
    )

def get_heuristic_move(game: GomokuGame) -> tuple:
    """간단한 휴리스틱 수 선택 (모델 없을 때 대체)"""
    import random
    
    valid_moves = game.get_valid_moves()
    if not valid_moves:
        return None
    
    board = game.board
    board_size = game.board_size
    
    # 1. 즉시 승리 가능한 수 찾기
    for row, col in valid_moves:
        temp_board = board.copy()
        temp_board[row, col] = game.current_player
        if check_win_at_position(temp_board, row, col, game.current_player):
            return (row, col)
    
    # 2. 상대방의 승리 막기
    opponent = -game.current_player
    for row, col in valid_moves:
        temp_board = board.copy()
        temp_board[row, col] = opponent
        if check_win_at_position(temp_board, row, col, opponent):
            return (row, col)
    
    # 3. 중앙 근처에 착수 (가중치 부여)
    center = board_size // 2
    weighted_moves = []
    for row, col in valid_moves:
        distance = abs(row - center) + abs(col - center)
        weight = 1.0 / (1.0 + distance * 0.2)
        weighted_moves.append(((row, col), weight))
    
    # 가중치에 따라 확률적으로 선택
    total_weight = sum(w for _, w in weighted_moves)
    r = random.uniform(0, total_weight)
    cumulative = 0
    for move, weight in weighted_moves:
        cumulative += weight
        if r <= cumulative:
            return move
    
    return random.choice(valid_moves)

def check_win_at_position(board, row, col, player):
    """특정 위치에서 승리 조건 확인"""
    board_size = len(board)
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    for dx, dy in directions:
        count = 1
        for direction in [1, -1]:
            x, y = row + dx * direction, col + dy * direction
            while (0 <= x < board_size and 0 <= y < board_size 
                   and board[x, y] == player):
                count += 1
                x += dx * direction
                y += dy * direction
        if count >= 5:
            return True
    return False

@router.post("/move", response_model=MoveResponse)
async def ai_move(request: MoveRequest):
    """
    AI의 다음 수 계산
    
    - **game_id**: 게임 세션 ID (선택)
    - **board**: 현재 보드 상태 (game_id 없을 때 필수)
    - **current_player**: 현재 플레이어 (board와 함께 사용)
    - **difficulty**: easy/normal/hard
    """
    
    # 게임 상태 가져오기 또는 생성
    if request.game_id and request.game_id in game_sessions:
        game = game_sessions[request.game_id]
    elif request.board is not None:
        game = GomokuGame(board_size=BOARD_SIZE)
        game.board = np.array(request.board, dtype=np.int8)
        game.current_player = request.current_player if request.current_player else 1
    else:
        raise HTTPException(status_code=400, detail="Either game_id or board must be provided")
    
    # 게임 종료 확인
    game_over, winner = game.is_game_over()
    if game_over:
        return MoveResponse(
            game_id=request.game_id or "stateless",
            row=-1,
            col=-1,
            board=game.board.tolist(),
            winner=int(winner) if winner is not None else None,
            game_over=True,
            message="Game already ended"
        )
    
    # 난이도에 따른 시뮬레이션 횟수 조정
    if request.difficulty == "easy":
        mcts.num_simulations = 100
    elif request.difficulty == "normal":
        mcts.num_simulations = 400
    else:  # hard
        mcts.num_simulations = 800
    
    # AI 수 계산
    try:
        if model_loaded:
            # MCTS + 신경망 사용
            move, _ = mcts.search(game, temperature=0.1)
            row, col = move
        else:
            # 모델 없을 때: 간단한 휴리스틱 사용
            move = get_heuristic_move(game)
            if move is None:
                return MoveResponse(
                    game_id=request.game_id or "stateless",
                    row=-1,
                    col=-1,
                    board=game.board.tolist(),
                    winner=0,
                    game_over=True,
                    message="No valid moves"
                )
            row, col = move
        
        # 착수
        success = game.make_move(row, col)
        if not success:
            raise HTTPException(status_code=500, detail="Invalid move generated by AI")
        
        # 승자 확인
        game_over, winner = game.is_game_over()
        
        message = f"AI plays at ({row}, {col})"
        if game_over:
            if winner == 1:
                message += " - Black wins!"
            elif winner == -1:
                message += " - White wins!"
            else:
                message += " - Draw!"
        
        return MoveResponse(
            game_id=request.game_id or "stateless",
            row=int(row),
            col=int(col),
            board=game.board.tolist(),
            winner=int(winner) if winner is not None else None,
            game_over=bool(game_over),
            message=message
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating move: {str(e)}")

@router.post("/human-move")
async def human_move(game_id: str, row: int, col: int):
    """사람의 수를 입력"""
    if game_id not in game_sessions:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = game_sessions[game_id]
    
    # 착수
    success = game.make_move(row, col)
    if not success:
        raise HTTPException(status_code=400, detail="Invalid move")
    
    # 승자 확인
    game_over, winner = game.is_game_over()
    
    return {
        "success": True,
        "board": game.board.tolist(),
        "current_player": int(game.current_player),  # numpy.int8 -> int
        "game_over": bool(game_over),  # numpy.bool_ -> bool
        "winner": int(winner) if winner is not None else None  # numpy.int8 -> int
    }

@router.get("/model-status")
async def get_model_status():
    """모델 상태 확인"""
    return {
        "model_loaded": model_loaded,
        "model_path": MODEL_PATH,
        "board_size": BOARD_SIZE,
        "status": "ready" if model_loaded else "not trained"
    }

@router.get("/info")
async def get_game_info():
    """게임 정보"""
    return {
        "name": "Gomoku (오목)",
        "board_size": f"{BOARD_SIZE}x{BOARD_SIZE}",
        "rules": "5개를 연속으로 놓으면 승리",
        "players": {
            "1": "Black (흑)",
            "-1": "White (백)"
        },
        "ai_algorithm": "MCTS + Deep Neural Network",
        "training": "Self-play reinforcement learning"
    }
