def linear(x):
    linear_grad = np.ones(x.shape)
    return x, linear_grad

def sigmoid(x):
    sigmoid = 1 / (1 + np.exp(-x))
    sigmoid_grad = sigmoid * (1 - sigmoid)
    return sigmoid, sigmoid_grad

def relu(x):
    relu = np.maximum(x, np.zeros(x.shape))
    relu_grad = np.ones(x.shape) * (x > 0)
    return relu, relu_grad


def new_layer(dim_in, dim_out):
    weights = np.random.rand(dim_out, dim_in)-0.5
    bias = np.random.rand(dim_out, 1)-0.5
    #bias = np.zeros((dim_out,1))
    return {
        'weights': weights,
        'bias': bias,
        'activations': np.zeros((dim_out, AEdata.shape[1])),
        'act_grad': np.zeros((dim_out, AEdata.shape[1])),
        'errors': np.zeros((dim_out, AEdata.shape[1])),
        'weights_grad': np.zeros(weights.shape),
        'bias_grad': np.zeros(bias.shape)
    }

def initialize_model():
    return [new_layer(29, 14),new_layer(14, 29)]

def forward_propagate(layers, inputs, activation_function):
    for layer in layers:
        zs = np.matmul(layer['weights'], inputs) + layer['bias']
        activations, act_grad = activation_function(zs)
        layer['activations'] = activations
        layer['act_grad'] = act_grad
        inputs = activations
        

def backward_propagate(layers, inputs, outputs):
    errors = (layers[-1]['activations'] - outputs)
    for layer_number, layer in reversed(list(enumerate(layers))):
        layer['errors'] = errors
        h = errors * layer['act_grad']
        if layer_number == 0:
            last_activations = inputs
        else:
            last_activations = layers[layer_number - 1]['activations']
        weights_grad = np.matmul(h, last_activations.transpose()) / inputs.shape[1]
        layer['weights_grad'] = weights_grad
        bias_grad = h.sum(axis=1).reshape(layer['bias'].shape) / inputs.shape[1]
        layer['bias_grad'] = bias_grad
        if layer_number != 0:
            errors = np.matmul(layer['weights'].transpose(), h)
            
            
def gradient_descent(layers, learning_rate):
    for layer in layers:
        layer['weights'] = layer['weights'] - layer['weights_grad'] * learning_rate
        layer['bias'] = layer['bias'] - layer['bias_grad'] * learning_rate
        
        
        
def error_function(layers):
    errors = layers[-1]['errors']
    e_squared = errors ** 2
    return e_squared.sum() / errors.shape[1]



def trainAE(activation_function, learning_rate, epochs,AEdata):
    layers = initialize_model()
    error_history = []
    for epoch in range(epochs):
        forward_propagate(layers, AEdata, activation_function)
        backward_propagate(layers, AEdata, AEdata)
        gradient_descent(layers, learning_rate)
        error_history.append(error_function(layers))
    
    return layers, error_history