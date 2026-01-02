"""
오목 게임 로직
15x15 보드에서 5개를 연속으로 놓으면 승리
"""
import numpy as np
from typing import Tuple, List, Optional

class GomokuGame:
    def __init__(self, board_size: int = 15):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player = 1  # 1: 흑돌, -1: 백돌
        self.move_history = []
        
    def reset(self):
        """게임 초기화"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        self.move_history = []
        return self.board.copy()
    
    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """가능한 수를 반환"""
        return [(i, j) for i in range(self.board_size) 
                for j in range(self.board_size) if self.board[i, j] == 0]
    
    def make_move(self, row: int, col: int) -> bool:
        """착수 실행"""
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return False
        if self.board[row, col] != 0:
            return False
        
        self.board[row, col] = self.current_player
        self.move_history.append((row, col, self.current_player))
        self.current_player = -self.current_player
        return True
    
    def check_winner(self, row: int, col: int) -> Optional[int]:
        """승자 확인 (마지막 착수 위치 기준)"""
        if self.board[row, col] == 0:
            return None
        
        player = self.board[row, col]
        directions = [
            (0, 1),   # 가로
            (1, 0),   # 세로
            (1, 1),   # 대각선 \
            (1, -1),  # 대각선 /
        ]
        
        for dx, dy in directions:
            count = 1
            # 양방향 체크
            for direction in [1, -1]:
                x, y = row + dx * direction, col + dy * direction
                while (0 <= x < self.board_size and 0 <= y < self.board_size 
                       and self.board[x, y] == player):
                    count += 1
                    x += dx * direction
                    y += dy * direction
            
            if count >= 5:
                return player
        
        return None
    
    def is_game_over(self) -> Tuple[bool, Optional[int]]:
        """게임 종료 여부 및 승자 반환"""
        # 마지막 착수 확인
        if self.move_history:
            last_row, last_col, _ = self.move_history[-1]
            winner = self.check_winner(last_row, last_col)
            if winner is not None:
                return True, winner
        
        # 무승부 (보드가 꽉 참)
        if len(self.get_valid_moves()) == 0:
            return True, 0
        
        return False, None
    
    def get_board_state(self) -> np.ndarray:
        """현재 보드 상태 반환"""
        return self.board.copy()
    
    def get_state_for_network(self) -> np.ndarray:
        """신경망 입력용 상태 (3채널)"""
        # 채널 0: 현재 플레이어의 돌
        # 채널 1: 상대 플레이어의 돌
        # 채널 2: 현재 플레이어 (흑/백)
        state = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        state[0] = (self.board == self.current_player).astype(np.float32)
        state[1] = (self.board == -self.current_player).astype(np.float32)
        state[2] = np.full((self.board_size, self.board_size), 
                          1.0 if self.current_player == 1 else 0.0)
        return state
    
    def clone(self):
        """게임 상태 복사"""
        new_game = GomokuGame(self.board_size)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.move_history = self.move_history.copy()
        return new_game
    
    def __str__(self) -> str:
        """보드 출력"""
        symbols = {0: '·', 1: '●', -1: '○'}
        lines = []
        lines.append('  ' + ' '.join([f'{i:2d}' for i in range(self.board_size)]))
        for i, row in enumerate(self.board):
            line = f'{i:2d} ' + ' '.join([symbols[cell] + ' ' for cell in row])
            lines.append(line)
        return '\n'.join(lines)
