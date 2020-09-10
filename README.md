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

Step-1 → Randomly initialize the weights to small numbers close to 0 but not 0.
Step-2 → Input the first observation of your dataset in the input layer, each feature in one node.
Step-3 → Forward-Propagation: From left to right, the neurons are activated in a way that the impact of each neuron's activation is limited by the weights. Propagate the activations until getting the predicted value.
Step-4 → Compare the predicted result to the actual result and measure the generated error(Cost function).
Step-5 → Back-Propagation: from right to left, the error is backpropagated. Update the weights according to how much they are responsible for the error. The learning rate decides how much we update weights.
Step-6 → Repeat step-1 to 5 and update the weights after each observation(Reinforcement Learning)
Step-7 → When the whole training set passed through the ANN, that makes and epoch. Redo more epochs.
