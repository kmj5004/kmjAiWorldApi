import random
import os

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from src.mnist.config_utils import load_hyperparams

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

path = './trained_data/mnist_net.pth'

class Network(nn.Module):
    def __init__(self, sizes):
        """입력층, 은닉층, 출력층을 초기화 하는 함수"""
        super(Network, self).__init__()
        self.num_layers = len(sizes)
        self.sizes = sizes

        layers = []
        for i in range(self.num_layers - 1): # 길이가 4인 리스트의 각 요소에 다 접근하려면 1 빼줘야함 (이것도 적을까)
            layers.append(nn.Linear(sizes[i], sizes[i + 1])) # 입력층(i번 인덱스의 뉴런들)에서 출력층(i+1번 인덱스의 뉴런들)로 연산하는 계층을 생성한 후 추가
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(p=0.4) # 드롭아웃 레이어 정의

        self.to(device)

    def forward(self, inputs):
        """입력(inputs)에 대한 신경망의 출력을 반환. 순전파 함수"""
        for i in range(self.num_layers - 2): # 마지막 계층을 제외한 모든 계층에 시그모이드 활성화 함수 적용. 시그모이드(활성화) 함수 -> 모든 실수값을 0 ~ 1로 압축시키는 함수
            inputs = torch.relu(self.layers[i](inputs))
            inputs = self.dropout(inputs) # 은닉층에 대하여 드롭아웃 적용
        inputs = self.layers[-1](inputs) # 출력 계층에는 활성화 함수 적용 안함
        return inputs

    def MBGD(self, training_data, test_data=None, eta=None, l2_lambda=None, progress_callback=None, stop_flag=None):
        """미니배치 확률적 경사 하강법을 사용하여 신경망 학습
           test_data가 입력되면 매 에포크 후 테스트 데이터에 대해 신경망을 평가함
           progress_callback: 학습 진행상황을 전달받을 콜백 함수
           stop_flag: 학습 중지 플래그 (딕셔너리)"""
        print("학습을 시작합니다.")
        hyperparameters = load_hyperparams()
        
        # None이 아닌 경우에만 사용, None이면 best_config에서 로드한 값 사용
        eta = eta if eta is not None else hyperparameters.eta
        l2_lambda = l2_lambda if l2_lambda is not None else hyperparameters.l2_lambda
        epochs = hyperparameters.epochs
        mini_batch_size = hyperparameters.mini_batch_size
        
        print(f"사용할 하이퍼파라미터: eta={eta}, l2_lambda={l2_lambda}, epochs={epochs}, mini_batch_size={mini_batch_size}")

        n = len(training_data)
        optimizer = optim.Adam(self.parameters(), lr=eta, weight_decay=l2_lambda)
        criterion = nn.CrossEntropyLoss()

        for j in range(epochs):
            if stop_flag and stop_flag.get("stop", False):
                print(f"학습이 중지되었습니다. (Epoch {j+1}에서 중지)")
                return False
            
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size) # 미니배치의 크기만큼 전체 훈련 데이터셋을 슬라이스하여 미니배치들을 준비
            ]

            epoch_loss = 0.0
            for mini_batch in mini_batches:
                if stop_flag and stop_flag.get("stop", False):
                    print(f"학습이 중지되었습니다. (Epoch {j+1}에서 중지)")
                    return False
                
                inputs = torch.tensor(np.array([x.ravel() for x, _ in mini_batch]), dtype=torch.float32).to(device) # 미니배치 데이터 준비
                labels = torch.tensor([np.argmax(y) for x, y in mini_batch], dtype=torch.long).to(device)

                optimizer.zero_grad() # 매 에포크마다 기울기 초기화
                outputs = self.forward(inputs) # 순전파를 수행하여 예측값 계산
                loss = criterion(outputs, labels) # 손실값 계산
                loss.backward() # 역전파를 수행하여 가중차에 대한 손실을 계산
                optimizer.step() # 계산된 기울기를 이용하여 가중치를 업데이트
                
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(mini_batches)
            
            if test_data:
                n_test = len(test_data)
                accuracy = self.evaluate(test_data)
                accuracy_percentage = 100 * (accuracy / n_test)
                print(f"Epoch {j+1}: {n_test} test examples; 정확도: {accuracy_percentage:.1f}%")
                
                if progress_callback:
                    progress_callback({
                        "epoch": j + 1,
                        "total_epochs": epochs,
                        "loss": avg_loss,
                        "accuracy": accuracy,
                        "accuracy_percentage": accuracy_percentage,
                        "total_test": n_test
                    })
            else:
                print(f"Epoch {j+1} 완료")
                if progress_callback:
                    progress_callback({
                        "epoch": j + 1,
                        "total_epochs": epochs,
                        "loss": avg_loss
                    })
        
        print("학습이 완료되었습니다")
        return True

    def evaluate(self, test_data):
        """신경망이 올바른 결과를 출력하는 테스트 입력의 수를 반환"""
        self.eval() # 평가 모드로 전환하여 훈련에만 사용되는 기능을 비활성화

        correct = 0
        with torch.no_grad(): # 평가에는 기울기 계산이 필요 없으므로 비활성화
            for x, y in test_data:
                inputs = torch.tensor(x.ravel(), dtype=torch.float32).to(device)
                label = y

                outputs = self.forward(inputs) # 순전파를 수행하여 예측값 획득
                pred = outputs.argmax(dim=0, keepdim=True) # 가장 큰 값을 예측값으로 저장
                if pred.item() == label: # 값이 가장 큰 뉴런과 정답이 일치하는지 확인
                    correct += 1
            self.train()

            return correct

    def save_model(self, path_to_save: str = path):
        os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
        torch.save(self.state_dict(), path_to_save)
        print(f"모델이 {path_to_save}에 저장되었습니다.")

    def load_model(self, path_to_load: str = path):
        try:
            self.load_state_dict(torch.load(path_to_load))
            self.eval()
            print("모델이 성공적으로 로딩되었습니다.")
        except FileNotFoundError:
            print("모델 파일을 찾는데 실패했습니다.")
            print("모델 학습을 먼저 진행해주세요.")


