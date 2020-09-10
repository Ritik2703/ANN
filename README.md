# Hare krishna
# ANN - 
Artificial Neural Networks or ANN is an information processing paradigm that is inspired by the way the biological nervous system such as brain process information. It is composed of large number of highly interconnected processing elements(neurons) working in unison to solve a specific problem.

# Implementation of ANN

# Tutorial - http://pypr.sourceforge.net/ann.html
# blog - https://towardsdatascience.com/introduction-to-artificial-neural-networks-ann-1aea15775ef9

# How Neural Networks Works-

A neural network is trained by adjusting neuron input weights based on the network's performance on example inputs. If the network classifies an image correctly, weights contributing to the correct answer are increased, while other weights are decreased. If the network misclassifies an image, the weights are adjusted in the opposite direction.

# Neuron -

![biological](https://miro.medium.com/max/488/0*EqHnlkHI-Ny_O5VH.png)

# vs

![Perceptron](https://miro.medium.com/max/875/0*2AMCbOiRQfpOmmkn.png)

Weight shows the strength of a particular node.
b is a bias value. A bias value allows you to shift the activation function up or down.
In the simplest case, these products are summed, fed to a transfer function (activation function) to generate a result, and this result is sent as output.
Mathematically, x1.w1 + x2.w2 + x3.w3 ...... xn.wn = ∑ xi.wi
Now activation function is applied 𝜙(∑ xi.wi)

# Activation function-
Activation function decides whether a neuron should be activated or not by calculating the weighted sum and further adding bias to it. The motive is to introduce non-linearity into the output of a neuron.

If we do not apply activation function then the output signal would be simply linear function(one-degree polynomial). Now, a linear function is easy to solve but they are limited in their complexity, have less power. Without activation function, our model cannot learn and model complicated data such as images, videos, audio, speech, etc.

## Types of Activation Functions:

### 1. Threshold Activation Function — (Binary step function)
![thres](https://miro.medium.com/max/574/0*U0N4CpMEpq1Suwxq.png)

### 2. Sigmoid Activation Function - (logistic Function)
![sigmoid](https://miro.medium.com/max/606/0*Q8OJJc7t3VRofJxu.png)

The drawback of the Sigmoid activation function is that it can cause the neural network to get stuck at training time if strong negative input is provided.

### 3. Hyperbolic Tangent Function - (tanh)
![tanh](https://miro.medium.com/max/554/0*dnH9K_K4tlkNWz-p.png)

The main advantage of this function is that strong negative inputs will be mapped to negative output and only zero-valued inputs are mapped to near-zero outputs.,So less likely to get stuck during training.

### 4. Rectified Linear Units - (relu)
ReLu is the most used activation function in CNN and ANN which ranges from zero to infinity.[0,∞)

![relu](https://miro.medium.com/max/753/0*9s238ozjLeNyzubR)

It should only be applied to hidden layers of a neural network. So, for the output layer use softmax function for classification problem and for regression problem use a Linear function.
Here one problem is some gradients are fragile during training and can die. It causes a weight update which will make it never activate on any data point again. Basically ReLu could result in dead neurons.
To fix the problem of dying neurons, Leaky ReLu was introduced. So, Leaky ReLu introduces a small slope to keep the updates alive. Leaky ReLu ranges from -∞ to +∞.

#### Leaky Relu
![leaky](https://miro.medium.com/max/875/0*gHdGQI6WTjUIrKCz.jpeg)

# How does the Neural Netwoek Work-
![1](https://miro.medium.com/max/875/1*LCsVhwMv4Wyz70cn6lIFqw.png)

![2](https://miro.medium.com/max/739/1*ge_oFHV8gZjXDWcXppJjPQ.png)

# How do neural network learn?
![3](https://miro.medium.com/max/875/0*Kgh8JGsgz1ovUot0.png)

![4](https://miro.medium.com/proxy/1*mTTmfdMcFlPtyu8__vRHOQ.gif)

# Batch Gradient descent-

![5](https://miro.medium.com/max/875/1*R_PX3tpnEWxOPn_F_jJlKA.png)

# Stochastic Gradient Descent(SGD)-

![6](https://miro.medium.com/max/809/1*wqSBgyrgsu0ZyAOlhLftWw.png)


# Training ANN with Stochastic Gradient Descent

1. Randomly initialize the weights to small numbers close to 0 but not 0.
2. Input the first observation of your dataset in the input layer, each feature in one node.
3. Forward-Propagation: From left to right, the neurons are activated in a way that the impact of each neuron's activation is limited by the weights. Propagate the activations until getting the predicted value.
4. Compare the predicted result to the actual result and measure the generated error(Cost function).
5. Back-Propagation: from right to left, the error is backpropagated. Update the weights according to how much they are responsible for the error. The learning rate decides how much we update weights.
6. Repeat step-1 to 5 and update the weights after each observation(Reinforcement Learning
7. When the whole training set passed through the ANN, that makes and epoch. Redo more epochs.

# Code -
```
# Artificial Neural Network


# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

#Create dummy variables
geography=pd.get_dummies(X["Geography"],drop_first=True)
gender=pd.get_dummies(X['Gender'],drop_first=True)

## Concatenate the Data Frames

X=pd.concat([X,geography,gender],axis=1)

## Drop Unnecessary columns
X=X.drop(['Geography','Gender'],axis=1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units= 6, kernel_initializer = 'he_uniform',activation='relu',input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform',activation='relu'))
# Adding the output layer
classifier.add(Dense(units= 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'Adamax', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
model_history=classifier.fit(X_train, y_train,validation_split=0.33, batch_size = 10, epochs = 100)

# list all data in history

print(model_history.history.keys())
# summarize history for accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)

print(score)

```
# Hyperparameter - selecting best no of neuron, activation function, batch size, epochs

```


# Artificial Neural Network


# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

#Create dummy variables
geography=pd.get_dummies(X["Geography"],drop_first=True)
gender=pd.get_dummies(X['Gender'],drop_first=True)

## Concatenate the Data Frames

X=pd.concat([X,geography,gender],axis=1)

## Drop Unnecessary columns
X=X.drop(['Geography','Gender'],axis=1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

## Perform Hyperparameter Optimization

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, LeakyReLU, BatchNormalization, Dropout
from keras.activations import relu, sigmoid



def create_model(layers, activation):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_dim=X_train.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
            
    model.add(Dense(units = 1, kernel_initializer= 'glorot_uniform', activation = 'sigmoid')) # Note: no activation beyond this point
    
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    return model
    
model = KerasClassifier(build_fn=create_model, verbose=0)


layer = [(20), (40, 20), (45, 30, 15)]
activations = ['sigmoid', 'relu']
param_grid = dict(layers = layer, activation=activations, batch_size = [128, 256], epochs=[(30),(50),(100),(500),(1000)])
grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=5)

grid_result = grid.fit(X_train, y_train)

print("best score and params is ",grid_result.best_score_,grid_result.best_params_)

pred_y = grid.predict(X_test)
y_pred= (pred_y>0.5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred, y_test)

# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)

print(score)

```
