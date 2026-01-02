"""
오목 AI 학습 스크립트
자기대국을 통한 강화학습
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gomoku.self_play import train_loop

if __name__ == "__main__":
    print("=" * 60)
    print("오목 AI 자기대국 학습 시작")
    print("=" * 60)
    print("\n설정:")
    print("- 보드 크기: 15x15")
    print("- 알고리즘: MCTS + Deep Neural Network")
    print("- 학습 방법: Self-play reinforcement learning")
    print("- 모델: ResNet-style CNN")
    print("\n학습 중...")
    print("-" * 60)
    
    # 학습 시작
    # 초보적인 학습: 적은 iteration으로 시작
    train_loop(
        num_iterations=50,        # 50번 반복
        games_per_iteration=5,     # 반복당 5게임
        batch_size=32,
        train_steps_per_iteration=50,
        model_path='models/gomoku/gomoku_net.pth'
    )
    
    print("\n" + "=" * 60)
    print("학습 완료!")
    print("모델 저장 위치: models/gomoku/gomoku_net.pth")
    print("=" * 60)
    print("\n게임 플레이:")
    print("1. 서버 시작: uvicorn main:app --reload")
    print("2. 브라우저에서 http://localhost:8000/static/gomoku.html 접속")
