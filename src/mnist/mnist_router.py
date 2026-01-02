import torch
from fastapi import APIRouter, BackgroundTasks, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import numpy as np
from typing import Optional
import asyncio

from src.mnist import mnist_loader, network
from src.mnist.config_utils import load_layers, save_hyperparams, Hyperparams, load_hyperparams

router = APIRouter(prefix="/mnist", tags=["MNIST"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_data, validation_data, test_data = mnist_loader.load_data()
net = network.Network(load_layers())

# Try to load model, but don't fail if it doesn't match
try:
    net.load_model("./trained_data/mnist_net.pth")
except Exception as e:
    print(f"Warning: Could not load MNIST model: {e}")

class ImageData(BaseModel):
    image: list[float]

class TrainRequest(BaseModel):
    eta: Optional[float] = None
    l2_lambda: Optional[float] = None
    save_model: bool = True

class OptimizeRequest(BaseModel):
    initial_etas: list[float] = [0.009, 0.01, 0.02, 0.03, 0.05, 0.06]
    initial_lambdas: list[float] = [0.00001, 0.0001, 0.001]
    iterations: int = 5

# 학습 상태 추적
training_status = {"is_training": False, "message": ""}
optimization_status = {"is_optimizing": False, "message": ""}
stop_training_flag = {"stop": False}

# WebSocket 연결 관리
active_connections: list[WebSocket] = []

@router.get("/best-config")
async def get_best_config():
    """best_config.json에 저장된 최적의 하이퍼파라미터 조회"""
    try:
        params = load_hyperparams()
        return {
            "eta": params.eta,
            "l2_lambda": params.l2_lambda,
            "epochs": params.epochs,
            "mini_batch_size": params.mini_batch_size
        }
    except Exception as e:
        return {"error": f"설정 파일을 불러올 수 없습니다: {str(e)}"}

@router.websocket("/ws/training")
async def websocket_training(websocket: WebSocket):
    """학습 진행상황을 실시간으로 전송하는 WebSocket"""
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.remove(websocket)

async def broadcast_progress(data: dict):
    """모든 연결된 클라이언트에게 진행상황 전송"""
    disconnected = []
    for connection in active_connections:
        try:
            await connection.send_json(data)
        except:
            disconnected.append(connection)
    
    for conn in disconnected:
        if conn in active_connections:
            active_connections.remove(conn)

@router.post("/predict")
async def predict(image: ImageData):
    """이미지 예측 (학습 중에는 사용 불가)"""
    if training_status["is_training"]:
        return {"error": "모델이 현재 학습 중입니다. 학습이 완료된 후 다시 시도하세요."}
    
    image_array = np.array(image.image).reshape(784, 1)
    tensor_img = torch.tensor(image_array.ravel(), dtype=torch.float32).to(device)
    prediction = torch.argmax(net.forward(tensor_img)).item()
    return {"result": prediction}

@router.post("/reload")
async def reload_model():
    """저장된 모델을 다시 로드"""
    if training_status["is_training"]:
        return {"error": "모델이 현재 학습 중입니다."}
    
    try:
        net.load_model()
        return {"message": "모델이 성공적으로 재로드되었습니다."}
    except Exception as e:
        return {"error": f"모델 로드 실패: {str(e)}"}

@router.get("/test")
async def test_model():
    """테스트 데이터로 모델 평가"""
    if training_status["is_training"]:
        return {"error": "모델이 현재 학습 중입니다."}
    
    accuracy = net.evaluate(test_data)
    total = len(test_data)
    percentage = accuracy / total * 100
    
    return {
        "accuracy": accuracy,
        "total": total,
        "percentage": round(percentage, 2),
        "message": f"정확도: {percentage:.2f}%"
    }

def train_background(eta: Optional[float], l2_lambda: Optional[float], save: bool):
    """백그라운드 학습 함수"""
    try:
        training_status["is_training"] = True
        training_status["message"] = "학습이 진행 중입니다..."
        stop_training_flag["stop"] = False
        
        def progress_callback(data):
            """학습 진행상황을 WebSocket으로 전송"""
            asyncio.run(broadcast_progress(data))
        
        completed = net.MBGD(
            training_data=training_data, 
            test_data=test_data, 
            eta=eta, 
            l2_lambda=l2_lambda,
            progress_callback=progress_callback,
            stop_flag=stop_training_flag
        )
        
        if not completed:
            training_status["message"] = "학습이 사용자에 의해 중지됨"
            asyncio.run(broadcast_progress({
                "status": "stopped",
                "message": training_status["message"]
            }))
        else:
            if save:
                net.save_model()
                net.load_model()
                training_status["message"] = "학습 완료 및 모델 저장됨 (자동 재로드 완료)"
            else:
                training_status["message"] = "학습 완료"
            
            asyncio.run(broadcast_progress({
                "status": "completed",
                "message": training_status["message"]
            }))
    except Exception as e:
        training_status["message"] = f"학습 중 오류 발생: {str(e)}"
        asyncio.run(broadcast_progress({
            "status": "error",
            "message": training_status["message"]
        }))
    finally:
        training_status["is_training"] = False
        stop_training_flag["stop"] = False

@router.post("/train")
async def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    """모델 학습 (eta나 l2_lambda가 None이면 best_config.json에서 로드)"""
    if training_status["is_training"]:
        return {"error": "이미 학습이 진행 중입니다.", "status": training_status}
    
    best_params = load_hyperparams()
    eta_to_use = request.eta if request.eta is not None else best_params.eta
    l2_lambda_to_use = request.l2_lambda if request.l2_lambda is not None else best_params.l2_lambda
    
    background_tasks.add_task(train_background, request.eta, request.l2_lambda, request.save_model)
    
    return {
        "message": "학습이 백그라운드에서 시작되었습니다.",
        "eta": eta_to_use,
        "l2_lambda": l2_lambda_to_use,
        "save_model": request.save_model,
        "info": "None 값은 best_config.json에서 로드됩니다."
    }

@router.get("/train/status")
async def train_status():
    """학습 상태 확인"""
    return training_status

@router.post("/train/stop")
async def stop_training():
    """학습 중지"""
    if not training_status["is_training"]:
        return {"error": "현재 진행 중인 학습이 없습니다."}
    
    stop_training_flag["stop"] = True
    return {"message": "학습 중지 요청이 전송되었습니다."}

def optimize_background(initial_etas: list[float], initial_lambdas: list[float], iterations: int):
    """백그라운드 최적화 함수"""
    try:
        optimization_status["is_optimizing"] = True
        optimization_status["message"] = "최적값 탐색이 진행 중입니다..."
        
        import numpy as np
        
        train_data, val_data, test_data = mnist_loader.load_data()
        
        best_eta, best_lambda = initial_etas[0], initial_lambdas[0]
        best_percentage = 0.0
        
        etas_to_test = initial_etas
        lambdas_to_test = initial_lambdas
        
        all_results = []
        
        for i in range(iterations):
            asyncio.run(broadcast_progress({
                "optimization": True,
                "iteration": i + 1,
                "total_iterations": iterations,
                "message": f"{i + 1}차 탐색 시작",
                "etas_to_test": etas_to_test,
                "lambdas_to_test": lambdas_to_test
            }))
            
            if i > 0:
                range_factor = 0.8 ** i
                
                log_eta = np.log10(best_eta)
                log_eta_min = log_eta - 0.5 * range_factor
                log_eta_max = log_eta + 0.5 * range_factor
                etas_to_test = np.logspace(log_eta_min, log_eta_max, 4).tolist()
                
                log_lambda = np.log10(best_lambda)
                log_lambda_min = log_lambda - 0.5 * range_factor
                log_lambda_max = log_lambda + 0.5 * range_factor
                lambdas_to_test = np.logspace(log_lambda_min, log_lambda_max, 4).tolist()
            
            current_best_percentage = -1.0
            current_best_eta, current_best_lambda = best_eta, best_lambda
            
            combination_count = 0
            total_combinations = len(etas_to_test) * len(lambdas_to_test)
            
            for eta in etas_to_test:
                for lambda_ in lambdas_to_test:
                    combination_count += 1
                    
                    net_test = network.Network(load_layers())
                    net_test.MBGD(training_data=train_data, eta=eta, l2_lambda=lambda_, test_data=val_data)
                    
                    accuracy = net_test.evaluate(val_data)
                    total = len(val_data)
                    percentage = accuracy / total * 100
                    
                    result = {
                        "iteration": i + 1,
                        "eta": eta,
                        "lambda": lambda_,
                        "accuracy": accuracy,
                        "percentage": percentage,
                        "total": total,
                        "is_best": False
                    }
                    all_results.append(result)
                    
                    asyncio.run(broadcast_progress({
                        "optimization": True,
                        "optimization_result": result,
                        "iteration": i + 1,
                        "combination": combination_count,
                        "total_combinations": total_combinations,
                        "current_best_eta": current_best_eta,
                        "current_best_lambda": current_best_lambda,
                        "current_best_percentage": current_best_percentage
                    }))
                    
                    if percentage > current_best_percentage:
                        current_best_percentage = percentage
                        current_best_eta, current_best_lambda = eta, lambda_
            
            if current_best_percentage > best_percentage:
                best_percentage = current_best_percentage
                best_eta = current_best_eta
                best_lambda = current_best_lambda
            
            if i < iterations - 1:
                best_eta = current_best_eta
                best_lambda = current_best_lambda
        
        save_hyperparams(Hyperparams(eta=best_eta, l2_lambda=best_lambda, epochs=40, mini_batch_size=32))
        
        for result in all_results:
            if result["eta"] == best_eta and result["lambda"] == best_lambda:
                result["is_best"] = True
        
        optimization_status["message"] = f"최적값 탐색 완료: eta={best_eta:.6f}, lambda={best_lambda:.6f}, accuracy={best_percentage:.2f}%"
        
        asyncio.run(broadcast_progress({
            "optimization": True,
            "status": "completed",
            "best_eta": best_eta,
            "best_lambda": best_lambda,
            "best_percentage": best_percentage,
            "all_results": all_results,
            "message": optimization_status["message"]
        }))
        
    except Exception as e:
        optimization_status["message"] = f"최적화 중 오류 발생: {str(e)}"
        asyncio.run(broadcast_progress({
            "optimization": True,
            "status": "error",
            "message": optimization_status["message"]
        }))
    finally:
        optimization_status["is_optimizing"] = False

@router.post("/optimize")
async def optimize_hyperparameters(request: OptimizeRequest, background_tasks: BackgroundTasks):
    """최적의 학습률과 람다 값 찾기 (시간 소요가 큼)"""
    if optimization_status["is_optimizing"]:
        return {"error": "이미 최적화가 진행 중입니다.", "status": optimization_status}
    
    if training_status["is_training"]:
        return {"error": "모델이 현재 학습 중입니다. 학습이 완료된 후 시도하세요."}
    
    background_tasks.add_task(optimize_background, request.initial_etas, request.initial_lambdas, request.iterations)
    
    return {
        "message": "최적값 탐색이 백그라운드에서 시작되었습니다.",
        "warning": "이 작업은 상당한 시간과 자원을 소모할 수 있습니다.",
        "initial_etas": request.initial_etas,
        "initial_lambdas": request.initial_lambdas,
        "iterations": request.iterations
    }

@router.get("/optimize/status")
async def optimize_status():
    """최적화 상태 확인"""
    return optimization_status
