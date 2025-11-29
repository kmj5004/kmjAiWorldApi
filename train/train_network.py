from src import mnist_loader, network

training_data, validation_data, test_data = mnist_loader.load_data()

net = network.Network([784, 100, 30, 10])

print("훈련이 시작되었습니다.")
print(f"Training data: {len(training_data)}, Validation data: {len(validation_data)}")
net.MBGD(training_data=training_data, test_data=test_data)
print("훈련이 완료되었습니다.")

net.save_model()
print(f"모델이 저장되었습니다.")
