"""
오목 AI 간단한 학습 스크립트
빠르고 안정적인 학습
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from collections import deque
import random

from src.gomoku.game import GomokuGame
from src.gomoku.network import GomokuNet
from src.gomoku.self_play import Trainer

def simple_play_game(board_size=15):
    """간단한 자기대국 (랜덤 + 간단한 전략)"""
    game = GomokuGame(board_size)
    game.reset()
    
    game_data = []
    
    while True:
        state = game.get_state_for_network()
        valid_moves = game.get_valid_moves()
        
        if not valid_moves:
            break
        
        # 랜덤 정책 (균등 분포)
        policy = np.zeros(board_size * board_size)
        for row, col in valid_moves:
            idx = row * board_size + col
            policy[idx] = 1.0 / len(valid_moves)
        
        # 랜덤하게 수 선택
        move = random.choice(valid_moves)
        game_data.append((state, policy, None))
        game.make_move(*move)
        
        # 게임 종료 확인
        game_over, winner = game.is_game_over()
        if game_over:
            # 가치 할당
            result_data = []
            for i, (state, policy, _) in enumerate(game_data):
                player_at_move = 1 if i % 2 == 0 else -1
                
                if winner == 0:
                    value = 0.0
                elif winner == player_at_move:
                    value = 1.0
                else:
                    value = -1.0
                
                result_data.append((state, policy, value))
            
            return result_data
    
    return []

def simple_train_loop(num_iterations=100, games_per_iteration=20, 
                     model_path='models/gomoku/gomoku_net_simple.pth'):
    """간단한 학습 루프"""
    print("=" * 60)
    print("오목 AI 간단 학습 시작")
    print("=" * 60)
    print("이 방법은 MCTS 없이 빠르게 기본 전략을 학습합니다")
    print("나중에 MCTS로 강화할 수 있습니다")
    print("=" * 60)
    
    board_size = 15
    network = GomokuNet(board_size=board_size, num_channels=128, num_res_blocks=5)
    trainer = Trainer(network, board_size=board_size, lr=0.001)
    
    # 기존 모델 로드 시도
    trainer.load_model(model_path)
    
    for iteration in range(num_iterations):
        print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")
        
        # 자기대국
        print(f"Playing {games_per_iteration} games...")
        for game_num in range(games_per_iteration):
            game_data = simple_play_game(board_size)
            if game_data:
                trainer.add_game_data(game_data)
            
            if (game_num + 1) % 10 == 0:
                print(f"  {game_num + 1}/{games_per_iteration} games completed")
        
        # 학습
        print(f"Training on {len(trainer.replay_buffer)} samples...")
        total_loss = 0
        for _ in range(50):
            loss, _, _ = trainer.train_step(batch_size=32)
            total_loss += loss
        
        avg_loss = total_loss / 50
        print(f"Average loss: {avg_loss:.4f}")
        
        # 모델 저장
        if (iteration + 1) % 10 == 0:
            trainer.save_model(model_path)
            print(f"✓ Model saved at iteration {iteration + 1}")
    
    # 최종 저장
    trainer.save_model(model_path)
    print("\n" + "=" * 60)
    print("학습 완료!")
    print(f"모델 저장: {model_path}")
    print("=" * 60)

if __name__ == "__main__":
    print("간단하고 빠른 오목 AI 학습")
    print("MCTS 없이 기본 전략만 학습합니다 (10-20분)")
    print()
    
    simple_train_loop(
        num_iterations=100,
        games_per_iteration=20,
        model_path='models/gomoku/gomoku_net.pth'
    )
    
    print("\n다음 단계:")
    print("1. 이 모델로 게임 플레이 테스트")
    print("2. 괜찮으면 MCTS로 추가 학습")
    print("3. 서버: uvicorn main:app --reload")
