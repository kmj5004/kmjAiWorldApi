"""
자기대국 학습
AlphaGo Zero 스타일
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
from collections import deque
import random
import os

from src.gomoku.game import GomokuGame
from src.gomoku.network import GomokuNet
from src.gomoku.mcts import MCTS

class SelfPlay:
    """자기대국 학습 엔진"""
    def __init__(self, network: GomokuNet, board_size: int = 15):
        self.network = network
        self.board_size = board_size
        self.mcts = MCTS(network, num_simulations=400)
        
    def play_game(self, temperature_threshold: int = 15) -> List[Tuple]:
        """한 게임 진행 (자기대국)"""
        game = GomokuGame(self.board_size)
        game.reset()
        
        game_data = []  # (state, policy, value)
        move_count = 0
        
        while True:
            # Temperature 조정 (초반엔 탐색적, 후반엔 확정적)
            temperature = 1.0 if move_count < temperature_threshold else 0.1
            
            # 현재 상태
            state = game.get_state_for_network()
            
            # MCTS로 다음 수 결정
            move, visit_counts = self.mcts.search(game, temperature)
            
            # 정책 (방문 횟수를 확률로 변환)
            policy = visit_counts / np.sum(visit_counts)
            
            # 데이터 저장 (승자는 나중에 결정)
            game_data.append((state, policy, None))
            
            # 착수
            game.make_move(*move)
            move_count += 1
            
            # 게임 종료 확인
            game_over, winner = game.is_game_over()
            if game_over:
                # 각 상태의 가치 할당
                result_data = []
                for i, (state, policy, _) in enumerate(game_data):
                    # 해당 착수 시점의 플레이어가 이겼는지
                    player_at_move = 1 if i % 2 == 0 else -1
                    
                    if winner == 0:
                        value = 0.0  # 무승부
                    elif winner == player_at_move:
                        value = 1.0  # 승리
                    else:
                        value = -1.0  # 패배
                    
                    result_data.append((state, policy, value))
                
                return result_data

class Trainer:
    """신경망 학습"""
    def __init__(self, network: GomokuNet, board_size: int = 15, lr: float = 0.001):
        self.network = network
        self.board_size = board_size
        self.optimizer = optim.Adam(network.parameters(), lr=lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)
        
        # 리플레이 버퍼
        self.replay_buffer = deque(maxlen=10000)
        
    def add_game_data(self, game_data: List[Tuple]):
        """게임 데이터를 버퍼에 추가"""
        self.replay_buffer.extend(game_data)
    
    def train_step(self, batch_size: int = 32) -> Tuple[float, float, float]:
        """학습 스텝"""
        if len(self.replay_buffer) < batch_size:
            return 0.0, 0.0, 0.0
        
        # 배치 샘플링
        batch = random.sample(self.replay_buffer, batch_size)
        states, policies, values = zip(*batch)
        
        # 텐서 변환
        states = torch.FloatTensor(np.array(states)).to(self.device)
        target_policies = torch.FloatTensor(np.array(policies)).to(self.device)
        target_values = torch.FloatTensor(np.array(values)).unsqueeze(1).to(self.device)
        
        # Forward
        self.network.train()
        pred_policies, pred_values = self.network(states)
        
        # Loss 계산
        policy_loss = -torch.mean(torch.sum(target_policies * pred_policies, dim=1))
        value_loss = nn.MSELoss()(pred_values, target_values)
        total_loss = policy_loss + value_loss
        
        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), policy_loss.item(), value_loss.item()
    
    def save_model(self, path: str):
        """모델 저장"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'buffer_size': len(self.replay_buffer)
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """모델 로드"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model loaded from {path}")
            return True
        return False

def play_single_game(network, board_size):
    """단일 게임 플레이 (병렬 처리용)"""
    self_play = SelfPlay(network, board_size)
    return self_play.play_game()

def train_loop(num_iterations: int = 100, games_per_iteration: int = 10, 
               batch_size: int = 32, train_steps_per_iteration: int = 50,
               model_path: str = 'models/gomoku/gomoku_net.pth',
               num_workers: int = 4):
    """자기대국 학습 루프 (병렬 처리)"""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    
    board_size = 15
    network = GomokuNet(board_size=board_size, num_channels=128, num_res_blocks=5)
    trainer = Trainer(network, board_size=board_size, lr=0.001)
    
    # 기존 모델 로드 시도
    trainer.load_model(model_path)
    
    # CPU 코어 수에 맞춰 워커 수 조정
    max_workers = min(num_workers, multiprocessing.cpu_count())
    print(f"Using {max_workers} parallel workers")
    
    for iteration in range(num_iterations):
        print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")
        
        # 자기대국 수행 (병렬)
        print(f"Playing {games_per_iteration} games in parallel...")
        
        # 병렬로 게임 실행
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(play_single_game, network, board_size) 
                      for _ in range(games_per_iteration)]
            
            completed = 0
            for future in futures:
                try:
                    game_data = future.result(timeout=300)  # 5분 타임아웃
                    trainer.add_game_data(game_data)
                    completed += 1
                    if completed % 5 == 0:
                        print(f"  Completed {completed}/{games_per_iteration} games")
                except Exception as e:
                    print(f"  Game failed: {e}")
        
        print(f"  All {completed} games completed")
        
        # 학습
        print(f"Training on {len(trainer.replay_buffer)} samples...")
        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        
        for step in range(train_steps_per_iteration):
            total_loss, policy_loss, value_loss = trainer.train_step(batch_size)
            total_loss_sum += total_loss
            policy_loss_sum += policy_loss
            value_loss_sum += value_loss
        
        avg_total_loss = total_loss_sum / train_steps_per_iteration
        avg_policy_loss = policy_loss_sum / train_steps_per_iteration
        avg_value_loss = value_loss_sum / train_steps_per_iteration
        
        print(f"Loss - Total: {avg_total_loss:.4f}, Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f}")
        
        # 모델 저장
        if (iteration + 1) % 10 == 0:
            trainer.save_model(model_path)
            print(f"Model saved at iteration {iteration + 1}")
    
    # 최종 저장
    trainer.save_model(model_path)
    print("\nTraining completed!")

if __name__ == "__main__":
    print("Starting Gomoku AI self-play training...")
    train_loop(num_iterations=100, games_per_iteration=10)
