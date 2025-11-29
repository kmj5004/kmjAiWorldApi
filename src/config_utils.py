import json
from pydantic import BaseModel

file_path = '../best_config.json'

class Hyperparams(BaseModel):
    eta: float
    l2_lambda: float
    epochs: int
    mini_batch_size: int

def save_hyperparams(params: Hyperparams):
    data_to_save = params.model_dump()
    with open(file_path, 'w') as f:
        json.dump(data_to_save, f, indent=4)
    print(f"하이퍼파라미터가 {file_path}에 저장되었습니다")

def load_hyperparams():
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return Hyperparams(**data)
    except FileNotFoundError:
        return Hyperparams(
            eta=0.1, l2_lambda=0.001, epochs=40, mini_batch_size=32
        )