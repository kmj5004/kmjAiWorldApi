"""
오목 AI 병렬 학습 스크립트
멀티코어를 활용한 고속 학습
"""
import sys
import os
import torch.multiprocessing as mp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gomoku.parallel_train import parallel_train_loop

if __name__ == "__main__":
    # 멀티프로세싱 설정
    mp.set_start_method('spawn', force=True)
    
    print("=" * 70)
    print("오목 AI 병렬 자기대국 학습")
    print("=" * 70)
    print("\n설정:")
    print("- 보드 크기: 15x15")
    print("- 알고리즘: MCTS + Deep Neural Network")
    print("- 학습 방법: Parallel Self-play reinforcement learning")
    print("- 모델: ResNet-style CNN")
    print(f"- CPU 코어: 최대 {mp.cpu_count()}개 활용")
    print("\n병렬 처리로 학습 속도가 크게 향상됩니다!")
    print("-" * 70)
    
    # 학습 시작 (안정성 우선 설정)
    parallel_train_loop(
        num_iterations=50,           # 50번 반복
        games_per_iteration=8,       # 반복당 8게임 (안정성)
        batch_size=32,               # 배치 크기
        train_steps_per_iteration=50,
        model_path='models/gomoku/gomoku_net.pth',
        num_workers=4                # 4개 워커로 제한
    )
    
    print("\n" + "=" * 70)
    print("학습 완료!")
    print("모델 저장 위치: models/gomoku/gomoku_net.pth")
    print("=" * 70)
    print("\n게임 플레이:")
    print("1. 서버 시작: uvicorn main:app --reload")
    print("2. 브라우저에서 http://localhost:8000/static/gomoku.html 접속")
    print("\n이제 훨씬 강력한 AI와 대결할 수 있습니다! 🎮")
