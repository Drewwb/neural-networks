# test_neural_network.py

from FeedForward import Layer_Dense, Activation_ReLU, Activation_Softmax, Loss_CategoricalCrossentropy, Optimizer_SGD
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Generate dataset
X, y = spiral_data(samples=100, classes=3)

# Normalize the input data
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Create model
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

# Loss function
loss_function = Loss_CategoricalCrossentropy()
# Optimizer
optimizer = Optimizer_SGD(learning_rate=0.1, decay=1e-3, momentum=0.9)

# Training loop
for epoch in range(10001):
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Calculate loss
    loss = loss_function.calculate(activation2.output, y)

    # Calculate accuracy
    predictions = np.argmax(activation2.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}')

    # Backward pass
    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
