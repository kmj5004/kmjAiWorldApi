import numpy as np
import os
from torch.utils.data import random_split

from torchvision import datasets, transforms

# 프로젝트 루트 기준 절대 경로 사용
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, 'data')

def load_data(validation_ratio=0.1):
    """mnist 데이터셋을 다운로드받아 numpy 형식의 열벡터로 변환하여 반환하는 함수"""
    full_train_dataset = datasets.MNIST(
        root=data_path,
        train=True, # 다운로드 받을 데이터셋이 학습용인지 여부
        download=True, # 다운로드를 할건지. root 인자로 넣은 파일 경로에 데이터셋이 없으면 다운로드, 있으면 패스
        transform=transforms.ToTensor() # 이미지 데이터에 적용할 변환을 지정. pyTorch 텐서 형태로 변환되어 픽셀값이 0~1 사이의 값으로 정규화됨
    )

    test_dataset = datasets.MNIST(
        root=data_path,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    train_size = int((1 - validation_ratio) * len(full_train_dataset)) # 검증용 데이터의 비율을 뺀 만큼의 훈련용 데이터 갯수 확보
    val_size = len(full_train_dataset) - train_size
    train_dataset, validation_dataset = random_split( # 훈련용 데이터셋에서 훈련용과 검증용 데이터셋을 비율에 맞게 랜덤 분배
        full_train_dataset, [train_size, val_size]
    )

    train_data_list = [item for item in train_dataset]
    training_inputs = [np.reshape(item[0].numpy(), (784, 1)) for item in train_data_list]# 28x28 크기의 이미지를 784x1 크기의 열벡터로 변환 -> 입력층 레이어에서 수월하게 입력받기 위함
    training_results = [vectorized_result(item[1]) for item in train_data_list] # 숫자 레이블을 10x1의 열백터로 변환(예: y가 3일 때, [0, 0, 0, 1, 0, 0, 0, 0, 0], 참고: 레이블은 훈련 데이터의 정답을 의미함
    training_data = list(zip(training_inputs, training_results))  # 이미지의 픽셀값과 숫자 레이블을 zip함수를 사용하여 1대1 매칭시킴

    validation_data_list = [item for item in validation_dataset]
    validation_inputs = [np.reshape(item[0].numpy(), (784, 1)) for item in validation_data_list]
    validation_data = list(zip(validation_inputs, [item[1] for item in validation_data_list]))

    test_data_list = [item for item in test_dataset]
    test_inputs = [np.reshape(item[0].numpy(), (784, 1)) for item in test_data_list]
    test_data = list(zip(test_inputs, [item[1] for item in test_data_list])) # 테스트용 데이터의 레이블을 열벡터로 변환하지 않는 이유는 아래 차이를 보고 이해 부탁
    # 훈련용 데이터는 출력 레이어의 뉴런들과 레이블을 비교해서 손실을 계산함 -> 출력 레이어의 출력값이 열벡터 값으로 나오기에, 데이터를 미리 열벡터로 변환하여 계산을 용이하게 함
    # 반대로 테스트용 데이터는 단순히 출력 레이어의 출력값과 레이블의 값이 일치하는지만 보면 되기에 굳이 열벡터로 변환할 필요가 없음

    return training_data, validation_data, test_data

def vectorized_result(number):
    e = np.zeros((10, 1))
    e[number] = 1.0
    return e

# training_data, validation_data, test_data = load_data()
# print("학습용 데이터 -", len(training_data))
# print("검증용 데이터 -", len(validation_data))
# print("테스트용 데이터 -", len(test_data))