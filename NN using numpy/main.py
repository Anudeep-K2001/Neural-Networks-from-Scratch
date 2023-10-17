import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(X, W):
    X_with_bias = np.insert(X, 0, 1, axis=0)
    z = np.dot(W, X_with_bias)
    a = sigmoid(z)
    return a

def initialize_biased_weights(layers, seed):
    np.random.seed(seed)
    biased_weights = [np.random.randn(layers[i], layers[i-1] + 1) for i in range(1, len(layers))]
    for i in range(len(layers)-1):
        biased_weights[i][:, 0] = 0
    return biased_weights

def compute_error(A, Y):
    error = np.square(np.subtract(A, Y)).mean()
    return error

def compute_dw(X, Y, A, W, h, m):
    positive = compute_error(forward(X, W + h), Y)
    negative = compute_error(forward(X, W - h), Y)
    dw = (positive - negative) / (2 * h)
    return dw

def compute_db(X, Y, A, W, B, h, m):
    positive = compute_error(forward(X, W, B + h), Y)
    negative = compute_error(forward(X, W, B - h), Y)
    db = (positive - negative) / (2 * h)
    return db


def multi_layer_nn(X_train, Y_train, X_test, Y_test, layers, alpha, epochs, h=0.00001, seed=2):

    layers = [X_train.shape[0]] + layers
    biased_weights = initialize_biased_weights(layers, seed)
    mse_errors = []

    m = X_train.shape[1]

    for epoch in range(epochs):
        # Forward-Prop
        A = [X_train]
        for i in range(len(layers) - 1):
            a = forward(A[i], biased_weights[i])
            A.append(a)

        mse_errors.append(compute_error(A[-1], Y_train))
        dZ = A[-1] - Y_train
        #Back-Prop
        for i in reversed(range(len(layers) - 1)):
            dw = compute_dw(A[i], Y_train, dZ, biased_weights[i], h, m)
            biased_weights[i] -= alpha * dw

    A_test = [X_test]
    for i in range(len(layers) - 1):
        a_test = forward(A_test[i], biased_weights[i])
        A_test.append(a_test)
    
    return biased_weights, mse_errors, A_test[-1]