from src import mnist_loader, network

training_data, validation_data, test_data = mnist_loader.load_data()

net = network.Network([784, 30, 10])

print("Training started")
print(f"Training data: {len(training_data)}, Validation data: {len(validation_data)}")
net.MBGD(training_data=training_data, test_data=test_data)
print("Training finished")

model_path = "../trained_data/mnist_net.pth"
net.save_model(model_path)
print(f"Model saved to {model_path}")
