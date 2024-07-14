import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


class LIFNeuron:
    def __init__(self, tau=20.0, threshold=1.0, rest=0.0):
        self.tau = tau
        self.threshold = threshold
        self.rest = rest
        self.voltage = self.rest

    def reset(self):
        self.voltage = self.rest

    def forward(self, input_current, dt=1.0):
        # Update membrane potential
        dv = (-(self.voltage - self.rest) + input_current) / self.tau
        self.voltage += dv * dt

        # Check for spike
        if self.voltage >= self.threshold:
            self.voltage = self.rest
            return 1.0
        else:
            return 0.0


class SingleLayerSNN:
    def __init__(self, input_size, num_neurons):
        self.input_size = input_size
        self.num_neurons = num_neurons
        self.neurons = [LIFNeuron() for _ in range(num_neurons)]
        self.weights = np.random.randn(num_neurons, input_size) * 0.1
        self.biases = np.zeros(num_neurons)

    def reset(self):
        for neuron in self.neurons:
            neuron.reset()

    def forward(self, inputs, dt=1.0):
        spikes = np.zeros(self.num_neurons)
        for i, neuron in enumerate(self.neurons):
            input_current = np.dot(self.weights[i], inputs) + self.biases[i]
            spikes[i] = neuron.forward(input_current, dt)
        return spikes


num_epochs = 10
learning_rate = 1e-3
time_window = 100


input_size = 28 * 28
num_neurons = 10
snn = SingleLayerSNN(input_size, num_neurons)


for epoch in range(num_epochs):
    running_loss = 0.0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch [{epoch + 1}/{num_epochs}]')
    for i, (images, labels) in progress_bar:
        images = images.view(images.size(0), -1).numpy()
        labels = labels.numpy()

        for img, label in zip(images, labels):
            snn.reset()
            spikes = np.zeros(num_neurons)
            for t in range(time_window):
                spikes += snn.forward(img)

            target = np.zeros(num_neurons)
            target[label] = 1.0
            error = target - spikes / time_window

            for i in range(num_neurons):
                snn.weights[i] += learning_rate * error[i] * img
                snn.biases[i] += learning_rate * error[i]

            running_loss += np.sum(error ** 2)

        progress_bar.set_postfix(loss=running_loss / ((i + 1) * len(images)))

    correct = 0
    total = 0
    snn.reset()
    for images, labels in test_loader:
        images = images.view(images.size(0), -1).numpy()
        labels = labels.numpy()

        for img, label in zip(images, labels):
            spikes = np.zeros(num_neurons)
            for t in range(time_window):
                spikes += snn.forward(img)

            predicted = np.argmax(spikes)
            total += 1
            if predicted == label:
                correct += 1

    accuracy = correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Accuracy: {accuracy * 100:.2f}%')

print('Training completed')
