"""
몬테카를로 트리 탐색 (MCTS)
AlphaGo Zero 스타일 구현
"""
import numpy as np
import math
from typing import Optional, Tuple
from src.gomoku.game import GomokuGame

class MCTSNode:
    """MCTS 트리 노드"""
    def __init__(self, game_state: GomokuGame, parent=None, move: Optional[Tuple[int, int]] = None, prior: float = 0.0):
        self.game_state = game_state
        self.parent = parent
        self.move = move  # 이 노드에 도달한 수
        self.prior = prior  # 정책망이 예측한 확률
        
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        
    @property
    def value(self) -> float:
        """평균 가치"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, parent_visit_count: int, c_puct: float = 1.4) -> float:
        """UCB 점수 계산 (탐색-활용 균형)"""
        if self.visit_count == 0:
            u = c_puct * self.prior * math.sqrt(parent_visit_count)
        else:
            u = c_puct * self.prior * math.sqrt(parent_visit_count) / (1 + self.visit_count)
        
        return self.value + u
    
    def select_child(self, c_puct: float = 1.4):
        """UCB 점수가 가장 높은 자식 노드 선택"""
        return max(self.children.values(), 
                  key=lambda node: node.ucb_score(self.visit_count, c_puct))
    
    def expand(self, policy_probs: np.ndarray):
        """노드 확장 (자식 노드 생성)"""
        valid_moves = self.game_state.get_valid_moves()
        board_size = self.game_state.board_size
        
        for move in valid_moves:
            row, col = move
            move_idx = row * board_size + col
            prior = policy_probs[move_idx]
            
            # 새 게임 상태 생성
            new_game = self.game_state.clone()
            new_game.make_move(row, col)
            
            # 자식 노드 생성
            child = MCTSNode(new_game, parent=self, move=move, prior=prior)
            self.children[move] = child
        
        self.is_expanded = True
    
    def update(self, value: float):
        """역전파 (백업)"""
        self.visit_count += 1
        self.value_sum += value
        
        if self.parent is not None:
            self.parent.update(-value)  # 상대방 관점에서는 부호 반대

class MCTS:
    """몬테카를로 트리 탐색"""
    def __init__(self, network, num_simulations: int = 800, c_puct: float = 1.4):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        
    def search(self, game_state: GomokuGame, temperature: float = 1.0) -> Tuple[int, int]:
        """MCTS 탐색 실행"""
        root = MCTSNode(game_state.clone())
        
        # 시뮬레이션 반복
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # 1. Selection (선택)
            while node.is_expanded and node.children:
                node = node.select_child(self.c_puct)
                search_path.append(node)
            
            # 2. Evaluation (평가)
            game_over, winner = node.game_state.is_game_over()
            
            if game_over:
                # 게임 종료 상태
                if winner == 0:
                    value = 0.0  # 무승부
                else:
                    # 현재 플레이어 관점에서 가치
                    value = 1.0 if winner == node.game_state.current_player else -1.0
            else:
                # 신경망으로 평가
                state = node.game_state.get_state_for_network()
                policy_probs, value = self.network.predict(state)
                
                # 3. Expansion (확장)
                node.expand(policy_probs)
            
            # 4. Backup (역전파)
            for node in reversed(search_path):
                node.update(value)
                value = -value
        
        # 방문 횟수 기반으로 수 선택
        visit_counts = np.zeros(game_state.board_size * game_state.board_size)
        for move, child in root.children.items():
            row, col = move
            move_idx = row * game_state.board_size + col
            visit_counts[move_idx] = child.visit_count
        
        # Temperature에 따라 수 선택
        if temperature == 0:
            # 가장 많이 방문한 수 선택
            move_idx = np.argmax(visit_counts)
        else:
            # 확률적으로 선택
            probs = visit_counts ** (1.0 / temperature)
            probs = probs / np.sum(probs)
            move_idx = np.random.choice(len(probs), p=probs)
        
        row = move_idx // game_state.board_size
        col = move_idx % game_state.board_size
        
        return (row, col), visit_counts
    
    def get_action_probs(self, game_state: GomokuGame, temperature: float = 1.0) -> np.ndarray:
        """행동 확률 분포 반환"""
        _, visit_counts = self.search(game_state, temperature)
        
        if temperature == 0:
            probs = np.zeros_like(visit_counts)
            probs[np.argmax(visit_counts)] = 1.0
        else:
            probs = visit_counts ** (1.0 / temperature)
            probs = probs / np.sum(probs)
        
        return probs
