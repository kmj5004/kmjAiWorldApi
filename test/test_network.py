from src import mnist_loader, network

training_data, validation_data, test_data = mnist_loader.load_data()

net = network.Network([784, 30, 10])

model_path = "../trained_data/mnist_net.pth"
net.load_model(model_path)

print("Test started")
print(f"Test data: {len(test_data)}")
accuracy = net.evaluate(test_data)
total = len(test_data)
percentage = accuracy /total * 100
print(f"Test finished. Accuracy: {percentage:.2f}%")