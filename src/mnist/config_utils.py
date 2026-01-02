import json
import os
from pydantic import BaseModel

# 프로젝트 루트 디렉토리 기준 절대 경로 사용
# config_utils.py is at: src/mnist/config_utils.py
# We need to go up 2 levels to reach project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
file_path = os.path.join(BASE_DIR, 'best_config.json')
layer_file_path = os.path.join(BASE_DIR, 'layers.json')

class Hyperparams(BaseModel):
    eta: float
    l2_lambda: float
    epochs: int
    mini_batch_size: int

class LayerSizes(BaseModel):
    sizes: list[int]

def save_hyperparams(params: Hyperparams):
    data_to_save = params.model_dump()
    with open(file_path, 'w') as f:
        json.dump(data_to_save, f, indent=4)
    print(f"하이퍼파라미터가 {file_path}에 저장되었습니다")

def load_hyperparams(file_path_override=None):
    path = file_path_override if file_path_override else file_path
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return Hyperparams(**data)
    except FileNotFoundError:
        print("최적의 값을 불러오는데 실패했습니다. 임의의 값을 대신 반환합니다. find_eta_lamda.py를 실행하신 후 다시 불러오기를 시도해주세요")
        return Hyperparams(
            eta=0.1, l2_lambda=0.001, epochs=40, mini_batch_size=32
        )

def load_layers(layer_path=layer_file_path):
    try:
        with open(layer_path, 'r') as f:
            data = json.load(f)
        return LayerSizes(**data).sizes
    except FileNotFoundError:
        print("레이어 구성을 불러오는데 실패했습니다. 기본 구성을 대신 반환합니다.")
        return [784, 256, 128, 10]