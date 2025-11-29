from src import mnist_loader
from src.config_utils import save_hyperparams, Hyperparams
from src.network import Network
import numpy as np

file_path = '../best_config.json'


def find_optimal_eta_lambda(initial_etas, initial_lambdas, iterations=5):
    print("=" * 50)
    print("!!!주의!!!")
    print("최적의 학습률과 람다 계수값을 찾기 위한 함수입니다. 상당한 시간과 자원이 소모 될 수 있습니다.")
    print("=" * 50)

    train_data, val_data, test_data = mnist_loader.load_data()

    best_eta, best_lambda = initial_etas[0], initial_lambdas[0]
    best_percentage = 0.0

    etas_to_test = initial_etas
    lambdas_to_test = initial_lambdas

    for i in range(iterations):
        print(f"\n======== {i + 1}차 탐색 (범위 좁히기) 시작 ========")

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

        print(f"-> ETA 후보군: {np.round(etas_to_test, 5).tolist()}")
        print(f"-> LAMBDA 후보군: {np.round(lambdas_to_test, 5).tolist()}")

        for eta in etas_to_test:
            for lambda_ in lambdas_to_test:
                net_test = Network([784, 100, 30, 10])
                print("=" * 50)
                print(f"탐색 {i + 1}차: eta: {eta:.6f}, lambda: {lambda_:.6f}")
                print("=" * 50)
                print(f"Training data: {len(train_data)}, Validation data: {len(val_data)}")

                net_test.MBGD(training_data=train_data, eta=eta, l2_lambda=lambda_, test_data=val_data)

                accuracy = net_test.evaluate(val_data)
                total = len(val_data)
                percentage = accuracy / total * 100
                print(f"테스트 완료. 정확도: {percentage:.2f}%")

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

    print("\n" + "=" * 50)
    print("5차 정밀 탐색 최종 결과 ️")
    print(f"최적의 학습률: {best_eta:.6f}, 최적의 람다: {best_lambda:.6f}")
    print(f"최고의 정확도: {best_percentage:.2f}%")
    print("=" * 50)

    save_hyperparams(Hyperparams(eta=best_eta, l2_lambda=best_lambda, epochs=40, mini_batch_size=32))


find_optimal_eta_lambda(
    initial_etas=[0.01, 0.1, 1.0, 3.0],
    initial_lambdas=[0.0001, 0.001, 0.01, 0.1]
)