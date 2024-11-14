import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('Assignment_4/data.csv', header=None, names=['x1', 'x2', 'label'])
X = np.array(data[['x1', 'x2']])
Y = np.array(data['label'])

# Initialize weights, bias, and learning rate
weights = np.random.rand(2)  # Initialize random weights
bias = np.random.rand()  # Initialize random bias
learning_rate = 0.1
num_epochs = 65

def step_function(t):
    return 1 if t >= 0 else 0

def predict(x):
    return step_function(np.dot(weights, x) + bias)

def plot_points(data):
    admitted = data[data['label'] == 1]
    rejected = data[data['label'] == 0]
    plt.scatter(admitted['x1'], admitted['x2'], color='blue', edgecolor='k', label="Admit")
    plt.scatter(rejected['x1'], rejected['x2'], color='red', edgecolor='k', label="Reject")
    plt.legend()

def plot_decision_boundary(weights, bias, color='g--'):
    x_vals = np.array([0, 1])
    y_vals = -(weights[0] * x_vals + bias) / weights[1]
    plt.plot(x_vals, y_vals, color)

def train_perceptron(X, Y, learning_rate=0.1, num_epochs=65):
    global weights, bias
    plt.figure(figsize=(8, 6))
    plot_points(data)
    plot_decision_boundary(weights, bias, 'r-')
    plt.savefig('Assignment_4/epoch_0.png')
    plt.clf()
    
    for epoch in range(num_epochs):
        for i in range(len(X)):
            x = X[i]
            y = Y[i]
            y_hat = predict(x)
            if y != y_hat:
                error = y - y_hat
                weights[0] += learning_rate * error * x[0]
                weights[1] += learning_rate * error * x[1]
                bias += learning_rate * error
        
        plot_points(data)
        plot_decision_boundary(weights, bias, 'g--')
        plt.title(f"Epoch {epoch + 1}")
        plt.savefig(f'Assignment_4/epoch_{epoch + 1}.png')
        plt.clf()
    
    plot_points(data)
    plot_decision_boundary(weights, bias, 'k-')
    plt.title("Final Boundary after Training")
    plt.savefig('Assignment_4/final_boundary.png')
    plt.show()

train_perceptron(X, Y)
