"""
병렬 자기대국 학습
멀티프로세싱을 활용한 고속 학습
"""
import numpy as np
import torch
import torch.multiprocessing as mp
from typing import List, Tuple
import os
import time

from src.gomoku.game import GomokuGame
from src.gomoku.network import GomokuNet
from src.gomoku.mcts import MCTS
from src.gomoku.self_play import Trainer

def worker_play_games(worker_id: int, num_games: int, network_state: dict, 
                      result_queue: mp.Queue, board_size: int = 15):
    """워커 프로세스: 게임 플레이"""
    try:
        # 네트워크 로드
        network = GomokuNet(board_size=board_size, num_channels=128, num_res_blocks=5)
        network.load_state_dict(network_state)
        network.eval()
        
        # MCTS 시뮬레이션 수 감소 (빠른 학습)
        mcts = MCTS(network, num_simulations=100)
        
        print(f"[Worker {worker_id}] Starting {num_games} games...", flush=True)
        
        for game_num in range(num_games):
            try:
                game = GomokuGame(board_size)
                game.reset()
                
                game_data = []
                move_count = 0
                max_moves = board_size * board_size  # 무한 루프 방지
                temperature_threshold = 15
                
                while move_count < max_moves:
                    temperature = 1.0 if move_count < temperature_threshold else 0.1
                    state = game.get_state_for_network()
                    
                    # MCTS 탐색
                    move, visit_counts = mcts.search(game, temperature)
                    policy = visit_counts / np.sum(visit_counts)
                    
                    game_data.append((state, policy, None))
                    game.make_move(*move)
                    move_count += 1
                    
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
                        
                        # 결과를 큐에 추가
                        result_queue.put(result_data)
                        print(f"[Worker {worker_id}] Game {game_num+1}/{num_games} done ({move_count} moves)", flush=True)
                        break
                
                if move_count >= max_moves:
                    print(f"[Worker {worker_id}] Game {game_num+1} timeout", flush=True)
                    
            except Exception as e:
                print(f"[Worker {worker_id}] Game {game_num+1} error: {e}", flush=True)
                continue
        
        print(f"[Worker {worker_id}] Completed all games", flush=True)
        
    except Exception as e:
        print(f"[Worker {worker_id}] Fatal error: {e}", flush=True)
        import traceback
        traceback.print_exc()

def parallel_train_loop(num_iterations: int = 50, 
                       games_per_iteration: int = 20,
                       batch_size: int = 64,
                       train_steps_per_iteration: int = 100,
                       model_path: str = 'models/gomoku/gomoku_net.pth',
                       num_workers: int = None):
    """병렬 자기대국 학습 루프"""
    
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)  # CPU 코어 수 - 1
    
    print(f"\n{'='*60}")
    print(f"병렬 오목 AI 학습")
    print(f"{'='*60}")
    print(f"Workers: {num_workers}")
    print(f"Iterations: {num_iterations}")
    print(f"Games per iteration: {games_per_iteration}")
    print(f"{'='*60}\n")
    
    board_size = 15
    network = GomokuNet(board_size=board_size, num_channels=128, num_res_blocks=5)
    trainer = Trainer(network, board_size=board_size, lr=0.001)
    
    # 기존 모델 로드 시도
    trainer.load_model(model_path)
    
    for iteration in range(num_iterations):
        iter_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")
        
        # 현재 네트워크 상태 가져오기
        network_state = network.state_dict()
        
        # 결과 큐
        result_queue = mp.Queue()
        
        # 게임 수를 워커들에게 분배
        games_per_worker = games_per_iteration // num_workers
        remaining_games = games_per_iteration % num_workers
        
        # 워커 프로세스 시작
        processes = []
        for worker_id in range(num_workers):
            num_games = games_per_worker + (1 if worker_id < remaining_games else 0)
            if num_games > 0:
                p = mp.Process(
                    target=worker_play_games,
                    args=(worker_id, num_games, network_state, result_queue, board_size)
                )
                p.start()
                processes.append(p)
        
        print(f"Started {len(processes)} worker processes")
        
        # 결과 수집
        collected_games = 0
        expected_games = games_per_iteration
        timeout_count = 0
        max_timeouts = 3
        
        print(f"Collecting game results...")
        while collected_games < expected_games and timeout_count < max_timeouts:
            try:
                game_data = result_queue.get(timeout=120)  # 2분 타임아웃
                trainer.add_game_data(game_data)
                collected_games += 1
                timeout_count = 0  # 성공하면 리셋
                
                if collected_games % 5 == 0 or collected_games == expected_games:
                    print(f"  Collected {collected_games}/{expected_games} games")
                    
            except:
                timeout_count += 1
                print(f"  Waiting for games... ({timeout_count}/{max_timeouts})")
                if timeout_count >= max_timeouts:
                    print(f"  Collection timeout - proceeding with {collected_games} games")
                    break
        
        # 모든 프로세스 종료 대기
        for p in processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        
        print(f"  Collected {collected_games} games in {time.time() - iter_start_time:.1f}s")
        
        # 학습
        print(f"\nTraining on {len(trainer.replay_buffer)} samples...")
        train_start_time = time.time()
        
        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        
        for step in range(train_steps_per_iteration):
            total_loss, policy_loss, value_loss = trainer.train_step(batch_size)
            total_loss_sum += total_loss
            policy_loss_sum += policy_loss
            value_loss_sum += value_loss
            
            if (step + 1) % 20 == 0:
                avg_loss = total_loss_sum / (step + 1)
                print(f"  Step {step+1}/{train_steps_per_iteration}, Loss: {avg_loss:.4f}")
        
        avg_total_loss = total_loss_sum / train_steps_per_iteration
        avg_policy_loss = policy_loss_sum / train_steps_per_iteration
        avg_value_loss = value_loss_sum / train_steps_per_iteration
        
        print(f"\nTraining completed in {time.time() - train_start_time:.1f}s")
        print(f"Loss - Total: {avg_total_loss:.4f}, Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f}")
        
        # 모델 저장
        if (iteration + 1) % 5 == 0:
            trainer.save_model(model_path)
            print(f"✓ Model saved at iteration {iteration + 1}")
        
        iter_time = time.time() - iter_start_time
        print(f"\nIteration {iteration + 1} completed in {iter_time:.1f}s")
    
    # 최종 저장
    trainer.save_model(model_path)
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    # 멀티프로세싱 설정
    mp.set_start_method('spawn', force=True)
    
    print("Starting parallel Gomoku AI training...")
    parallel_train_loop(
        num_iterations=50,
        games_per_iteration=20,
        batch_size=64,
        train_steps_per_iteration=100,
        num_workers=None  # 자동으로 CPU 코어 수 감지
    )
