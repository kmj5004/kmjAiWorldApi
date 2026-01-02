from src.mnist import mnist_loader, network
from src.mnist.config_utils import load_layers

training_data, validation_data, test_data = mnist_loader.load_data()

net = network.Network(load_layers())

net.load_model()

print("테스트가 시작되었습니다.")
print(f"Test data: {len(test_data)}")
accuracy = net.evaluate(test_data)
total = len(test_data)
percentage = accuracy /total * 100
print(f"테스트가 완료되었습니다. 정확도: {percentage:.2f}%")
