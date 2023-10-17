import numpy as np
import tensorflow as tf


class Model(tf.Module):
    '''
    x = input or X_train
    arch = layers of nodes
    activations = list of activations
    '''
    def __init__(self, x, arch, activations, seed):
        self.arch = arch
        self.activations = activations


        # Activation functions
        self.func_activation = {
            'sigmoid' : tf.math.sigmoid,
            'relu' : self.relu,
            'linear' : self.linear
        }

        # Loss functions
        self.func_loss = {
            'cross_entropy' : self.Cross_entropy,
            'mse' : self.MSE,
            'svm' : self.SVM
        }


        # Initiate weights and bias
        if type(arch[0]) == int:
            self.build(x, arch, seed)
        else:
            self.weights = []
            self.biases = []
            for c in arch:
                self.biases.append(c[0])
                self.weights.append(c[1:])



    # Relu activation function
    def relu(self, x):
        return tf.maximum(0, x)
    

    # Linear activation function
    def linear(self, x):
        return x


    # MSE
    def MSE(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))
    

    # Cross entropy
    def Cross_entropy(self, y_true, y_pred):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred))
    

    # SVM
    def SVM(self, y_true, y_pred):
        s = tf.losses.Hinge()
        return s(y_true, y_pred)




    # Initialize weights and biases
    def build(self, x, arch, seed):

        input_dim = x.shape[1]

        weights = []
        biases = []

        shape = [input_dim, arch[0]]

        init_w = np.random.randn
        init_b = np.random.randn

        # Init weights and bias
        for idx, layer in enumerate(arch[:-1]):
            np.random.seed(seed)
            biases.append(tf.Variable(init_b(1, shape[1]), dtype=x.dtype))
            weights.append(tf.Variable(init_w(shape[0], shape[1]), dtype=x.dtype))
            shape[0] = layer
            shape[1] = arch[idx+1]

        np.random.seed(seed)
        biases.append(tf.Variable(init_b(1, shape[1]), dtype=x.dtype))
        weights.append(tf.Variable(init_w(shape[0], shape[1]), dtype=x.dtype))

        self.weights = weights
        self.biases = biases



    # Feedforward
    def __call__(self, x):

        # tf.matmul(w, x) + b
        current = x
        for weight, bias, activation in zip(self.weights, self.biases, self.activations):
            # matmul
            z = tf.matmul(current, weight) + bias
            # passing through activation function
            current = self.func_activation[activation.lower()](z)

        return current


    # Compute loss
    def compute_loss(self, function, actual, pred):
        return self.func_loss[function.lower()](actual, pred)


# Gives output with weights and biases combined
def stack(weights, biases):
    return [np.insert(weight, 0, bias, axis=0) for weight, bias in zip(weights, biases)]


# Splits data into train and validation
def split(X, y, validation_split):

    data_size = X.shape[0]
    start = int(validation_split[0] * data_size)
    end = int(validation_split[1] * data_size)

    x_test = X[start:end]
    y_test = y[start:end]


    x_train = np.concatenate((X[:start], X[end:]))
    y_train = np.concatenate((y[:start], y[end:]))

    return x_train, x_test, y_train, y_test


# Main
def multi_layer_nn_tensorflow(X_train,Y_train,layers,activations,alpha,batch_size,epochs=1,loss="svm",
                              validation_split=[0.8,1.0],seed=2):
    
    # Split data into train, test
    X_tr, X_test, Y_tr, Y_test = split(X_train, Y_train, validation_split)

    #create model
    model = Model(X_tr, layers, activations, seed)


    # Create loop for epochs
    error = []
    for epoch in range(epochs):
        # Calculate gradients of forward pass with respect to loss
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = model(X_tr)
            # Calculate loss
            ls = model.compute_loss(loss, Y_tr, y_pred)        
            # Calculate gradients
            g_w, g_b = tape.gradient(ls, [model.weights, model.biases])
        
        # Apply gradients
        [w.assign_sub(gw * alpha) for w, gw in zip(model.weights, g_w)]
        [b.assign_sub(gb * alpha) for b, gb in zip(model.biases, g_b)]

        # Validation data
        y_test_pred = model(X_test)
        ls = model.compute_loss(loss, Y_test, y_test_pred)
        error.append(ls.numpy())

    return [stack(model.weights, model.biases), error, model(X_test)]

