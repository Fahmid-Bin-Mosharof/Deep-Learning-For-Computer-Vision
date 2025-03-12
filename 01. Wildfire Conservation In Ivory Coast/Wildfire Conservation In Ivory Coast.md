# Project Overview

In this project, you'll look at a data science competition helping scientists track animals in a wildlife preserve. The goal is to take images from camera traps and classify which animal, if any, is present. To complete the competition, you'll expand your machine learning skills by creating more powerful neural network models that can take images as inputs and classify them into one of multiple categories.

Some of the things you'll learn are:

How to read image files and prepare them for machine learning
How to use PyTorch to manipulate tensors and build a neural network model
How to build a Convolutional Neural Network that works well with images
How to use that model to make predictions on new images
How to turn those predictions into a submission to the competition

# Competition Description

https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation

# Neural Network

Neural networks are the fundamental building blocks of deep learning algorithms. A neural network is a type of machine learning algorithm that is designed to simulate the behavior of the human brain. It is made up of interconnected nodes, also known as artificial neurons, which are organized into layers.

![Alt Text](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*dVU2SuYaw6dZWYHnt2y13Q.png)

# How Neural Network works?

Each neuron represents a unit of computation that takes in a set of inputs, performs a set of calculations, and produces an output that is passed on to the next layer.

Just like the neurons in our brains, each node in a neural network receives input, processes it, and passes the output on to the next node. As the data moves through the network, the connections between the nodes are strengthened or weakened, depending on the patterns in the data. This allows the network to learn from the data and make predictions or decisions based on what it has learned.

Imagine a 28 by 28 grid, where a number is drawn in such a way that some pixels are darker than others. By identifying the brighter pixels, we can decipher the number that was written on the grid. This grid serves as the input for a neural network.

![Alt Text](https://3b1b-posts.us-east-1.linodeobjects.com/content/lessons/2017/neural-networks/pixel-values.png)

The rows of the grid are arranged in a horizontal 1-D array, which is then transformed into a vertical array, forming the first layer of neurons. Just like this;

![Alt Text](https://miro.medium.com/v2/resize:fit:640/format:webp/1*PPOrlyeO7fnWo_sXeFTTMg.gif)

In the case of the first layer, each neuron corresponds to a single pixel in the input image, and the value inside each neuron represents the activation or intensity of that pixel. The input layer of a neural network is responsible for taking in the raw data (in this case, an image) and transforming it into a format that can be processed by the rest of the network.

In this case, we have 28x28 input pixels, which gives us a total of 784 neurons in the input layer. Each neuron will have an activation value of either 0 or 1, depending on whether the corresponding pixel in the input image is black or white, respectively.

![Alt Text](https://miro.medium.com/v2/resize:fit:640/format:webp/0*AU1wW2CnRFWAd1am.gif)

The output layer of the neural network consists of 10 neurons in this case, each of which represents a possible output class (in this case, the digits 0 through 9). The output of each neuron in the output layer represents the probability that the input image belongs to that particular class. The highest probability value determines the predicted class for that input image.

# Hidden Layers

In between the input and output layers, we have one or more hidden layers, which perform a series of non-linear transformations on the input data. The purpose of these hidden layers is to extract higher-level features from the input data that are more meaningful for the task at hand. It is upto you how many hidden layers you want to add in your network.

![Alt Text](https://miro.medium.com/v2/resize:fit:640/format:webp/0*_oLtMBUfvBANOk5-.gif)

Each neuron in the hidden layer receives inputs from all neurons in the previous layer, and applies a set of weights and biases to those inputs before passing the result through a non-linear activation function. This process is repeated across all neurons in the hidden layer until the output layer is reached.

# Terminologies used in Neural Networks

- Training of the neural network is the process of adjusting the weights of a neural network based on input data and desired output, in order to improve the accuracy of the network’s predictions.

- Weight: weights refer to the parameters that are learned during training, and they determine the strength of the connections between neurons. Each connection between neurons is assigned a weight, which is multiplied by the input value to the neuron to determine its output.

![Alt Text](https://miro.medium.com/v2/resize:fit:640/format:webp/1*w1w091o_Cwdk0t_UvHApyQ.png)

- Bias: Bias is another learned parameter that is added to the weighted sum of inputs to a neuron in a given layer. It is an additional input to the neuron that helps to adjust the output of the activation function.
- Non-linear activation function: A non-linear activation function is applied to the output of a neuron to introduce non-linearity into the network. Non-linearity is important because it allows the network to model complex, nonlinear relationships between inputs and outputs. Common activation functions used in neural networks include the sigmoid function, the ReLU (Rectified Linear Unit) function, and the softmax function.
- Loss function: This is a mathematical function that measures the error or difference between the predicted output of the neural network and the true output. The empirical loss measures the total loss over our entire dataset. Cross-entropy loss is commonly used with models that output a probability between 0 and 1, while mean squared error loss is used with regression models that output continuous real numbers. The goal is to minimize the loss function during training in order to improve the accuracy of the network’s predictions.
- Loss optimization: This is the process of minimizing the error or loss incurred by the neural network when making predictions. This is done by adjusting the weights of the network.
- Gradient descent: This is an optimization algorithm used to find the minimum of a function, such as the loss function of a neural network. It involves iteratively adjusting the weights in the direction of the negative gradient of the loss function. The idea is to keep moving the weights in the direction that reduces the loss, until we reach the minimum.


# Gradi