import numpy as np

# Initialize parameters
def init_params():
    W1 = np.random.randn(2, 2)   # 2 neurons in hidden layer
    B1 = np.zeros((2, 1))
    W2 = np.random.randn(1, 2)   # 1 output neuron
    B2 = np.zeros((1, 1))
    return W1, B1, W2, B2

# Activation functions
def ReLU(Z):
    return np.maximum(0, Z)

def deri_ReLU(Z):
    return Z > 0

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

# Forward propagation
def forward_prop(W1, B1, W2, B2, X):
    Z1 = W1.dot(X) + B1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + B2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

# Backward propagation
def backward_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = X.shape[1]
    DZ2 = A2 - Y
    DW2 = (1 / m) * DZ2.dot(A1.T)
    DB2 = (1 / m) * np.sum(DZ2, axis=1, keepdims=True)
    DZ1 = W2.T.dot(DZ2) * deri_ReLU(Z1)
    DW1 = (1 / m) * DZ1.dot(X.T)
    DB1 = (1 / m) * np.sum(DZ1, axis=1, keepdims=True)
    return DW1, DB1, DW2, DB2

# Update parameters
def update_params(W1, B1, W2, B2, DW1, DB1, DW2, DB2, alpha):
    W1 -= alpha * DW1
    B1 -= alpha * DB1
    W2 -= alpha * DW2
    B2 -= alpha * DB2
    return W1, B1, W2, B2

# Predictions
def get_prediction(A2):
    return (A2 > 0.5).astype(int)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

# Training loop
def gradient_descent(X, Y, epoch, alpha):
    W1, B1, W2, B2 = init_params()
    for i in range(epoch):
        Z1, A1, Z2, A2 = forward_prop(W1, B1, W2, B2, X)
        DW1, DB1, DW2, DB2 = backward_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, B1, W2, B2 = update_params(W1, B1, W2, B2, DW1, DB1, DW2, DB2, alpha)
        
        if i % 100 == 0:
            print("Iteration:", i, "Accuracy:", get_accuracy(get_prediction(A2), Y))
    return W1, B1, W2, B2

# XOR dataset
X = np.array([[0,0,1,1], [0,1,0,1]])   # inputs
Y = np.array([[0,1,1,0]])              # XOR outputs

# Train
W1, B1, W2, B2 = gradient_descent(X, Y, epoch=500, alpha=0.1)

# Final predictions
_, _, _, A2 = forward_prop(W1, B1, W2, B2, X)
print("Final Predictions:", get_prediction(A2))
print("True Labels:", Y)
