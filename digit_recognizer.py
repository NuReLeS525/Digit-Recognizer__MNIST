import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

data_url = './train.csv' 
data = pd.read_csv(data_url) 
data 

Y_data = data['label'] 
X_data = data.drop('label', axis = 1) 

Y_data = np.array(Y_data) 
X_data = np.array(X_data) 
print(f'Examples X: {X_data.shape[0]},Pixels per Example X: {X_data.shape[1]}, Lables Y: {Y_data.shape[0]}') 

num_val = 1000 

X_val = X_data[:num_val] 
Y_val = Y_data[:num_val] 
X_val = X_val / 255 

X_train = X_data[num_val:].T 
Y_train = Y_data[num_val:] 
X_train = X_train / 255 

print('Val: ', X_val.shape, Y_val.shape) 
print('Train: ', X_train.shape, Y_train.shape) 

def initialize_parameters(input_size, hidden_size, output_size): 
    std_dev = np.sqrt(2 / input_size) 
 
    W1 = np.random.randn(hidden_size, input_size) * std_dev 
    b1 = np.zeros((hidden_size, 1)) 
    W2 = np.random.randn(output_size, hidden_size) * std_dev 
    b2 = np.zeros((output_size, 1)) 
    return W1, b1, W2, b2 

def softmax(Z): 
    return np.exp(Z) / sum(np.exp(Z)) 

def forward_propagation(X, W1, b1, W2, b2):
    print('Weights: ', W1.shape, 'Inputs: ', X.shape)
    Z1 = np.dot(W1, X) + b1 
    A1 = np.maximum(Z1, 0) 
    Z2 = np.dot(W2, A1) + b2 
    A2 = softmax(Z2) 
    return Z1, A1, Z2, A2 

def one_hot_encoder(Y, num_classes): 
    num_samples = len(Y) 
    one_hot_Y = np.zeros((num_samples, num_classes)) 
    one_hot_Y[np.arange(num_samples), Y] = 1 
    return one_hot_Y.T 

def backward_propagation(X, Y, Z1, A1, Z2, A2, W2): 
    m = X.shape[1] 
    one_hot_Y = one_hot_encoder(Y, A2.shape[0]) 
 
    dZ2 = A2 - one_hot_Y 
    dW2 = 1 / m * np.dot(dZ2, A1.T) 
    db2 = 1 / m * np.sum(dZ2) 
 
    dZ1 = np.dot(W2.T, dZ2) * (Z1 > 0) 
    dW1 = 1 / m * np.dot(dZ1, X.T) 
    db1 = 1 / m * np.sum(dZ1) 
 
    return dW1, db1, dW2, db2 

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate): 
    W1 -= learning_rate * dW1 
    b1 -= learning_rate * db1 
    W2 -= learning_rate * dW2 
    b2 -= learning_rate * db2 
    return W1, b1, W2, b2 

def get_predictions(A2): 
    return np.argmax(A2, 0) 

def get_accuracy(prediction, Y): 
    accuracy = np.sum(prediction == Y) / Y.size 
 
    return accuracy 

def gradient_descent(X, Y, input_size, hidden_size, output_size, learning_rate, iterations): 
    history = { 
        'accuracy': [] 
    } 
     
    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size) 
 
    for i in range(iterations): 
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2) 
         
        dW1, db1, dW2, db2 = backward_propagation(X, Y, Z1, A1, Z2, A2, W2) 
         
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate) 
         
        if i % 10 == 0: 
            prediction = get_predictions(A2) 
            accuracy = get_accuracy(prediction, Y) 
 
            print(f'Iteration {i}, Accuracy: {accuracy}') 
            history['accuracy'].append(accuracy) 
             
    return W1, b1, W2, b2, history 

input_size = 784 
hidden_size = 10 
output_size = 10 
learning_rate = 0.1 
iterations = 500 

W1, b1, W2, b2, history = gradient_descent(X_train, Y_train, input_size, hidden_size, output_size, learning_rate, iterations) 

pd.DataFrame(history['accuracy']).plot(title='Training  Accuracy') 

def predict(X, W1, b1, W2, b2): 
    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2) 
    predictions = get_predictions(A2) 
    return predictions 

def val_predict_image(X, Y, index, W1, b1, W2, b2): 
    image_pixels = X.T[:, index, None]
    print(f'Image pixels at index {index} ', image_pixels.shape)
    print( X.shape, X.T.shape)
    prediction = predict(image_pixels, W1, b1, W2, b2) 
 
    label = Y[index] 
    print("Prediction: ", prediction) 
    print("Label: ", label) 
 
    image_pixels = image_pixels.reshape((28, 28)) * 255 
    plt.gray() 
    plt.imshow(image_pixels, interpolation='nearest') 
    plt.show() 

val_predict_image(X_val, Y_val, 1, W1, b1, W2, b2)
val_predict_image(X_val, Y_val, 6, W1, b1, W2, b2)
val_predict_image(X_val, Y_val, 11, W1, b1, W2, b2)
val_predict_image(X_val, Y_val, 200, W1, b1, W2, b2)

predictions = predict(X_val.T, W1, b1, W2, b2)
get_accuracy(predictions, Y_val)

data_path = './test.csv'
X_test = pd.read_csv(data_path)
X_test = np.array(X_test).T
X_test = X_test / 255
print(X_test.shape)

predictions = predict(X_test, W1, b1, W2, b2)
predictions.shape

data_path = './sample_submission.csv'
sample_submission = pd.read_csv(data_path)
sample_submission.head()

sample_submission['Label'] = predictions
sample_submission.head()

sample_submission.to_csv('submission.csv', index = False)