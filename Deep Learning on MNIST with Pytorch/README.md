# Image-Classification-with-MLP-MNIST-dataset
In this Project, I classify handwritten digits from the MNIST database. The MNIST dataset is included in torchvision.datasets and can easily be imported and loaded. Using this dataset, we'll classify the digits  [0,9]  and develop a network architecture that includes ten neurons whose outputs represent the probability of the digits.

## 1 Mathematical Foundation

First, recall for **binary classification**, we developed a probabilistic interpretation for the output of the **sigmoid** activation function, $y'$, as follows:

$$P(y\ =\ 1| x;\theta) = y'$$

Here $y'$ represented the probability of class 1 given the input $x$ is associated with class 1. And therefore, the probability that the input sample belongs class 0 (or the negative class) is:

$$P(y\ =\ 0| x;\theta) = 1 - y'$$

We combined these two expressions into a single equation as shown below, where $y$ represents the ground truth (or label) for the class.

$$p(y\ |\ x;\theta) = (y')\ ^y\ (1 - y')^{1-y}$$

In order to extend this to three or more classes, we are going to introduce the **softmax** activation function in the next section.

### 1.1 Softmax Activation Function

For classification problems involving more than two classes the target $y$ is a variable that ranges over more than two classes, and we, therefore, want to know the probability of $y$ being in each potential class $i$.

$$ p(y\ =\ i\ |\ x) = y_i'$$

As we will see further below, when we have three or more classes, the network architecture will contain an output neuron for each class whose output will be the predicted probability that the input $x$ is associated with a particular class, $i$. We can now use a generalization of the sigmoid activation function called the **softmax** function to compute $p(y\ =\ i\ |\ x)$.

The softmax activation function will map each of the neuron inputs to a probability distribution, where each neuron output is a value in the range $(0, 1)$ with all values summing to 1. Assuming the number of possible classes is $k$, the following equation defines the **softmax** function, and the output for any particular neuron with the input $ z_i = w_i^Tx + b_i $ is defined as follows:

$$ softmax(z_i)\ = \frac{e^{z_i}}{\sum_{j=1}^{k} e^{z_j}} = y_i', \ \ 1<= i <= k $$

Therefore, each output neuron ($i$) will compute a softmax score according to the above equation. Notice that the numerator is for a given class $i$, and the denominator normalizes each neuron's output into probabilities so that the inputs are mapped to the range $(0,1)$. So each output neuron computes a number in the range $(0,1)$, and the summation of the scores from all neurons is 1. The output of each neuron, $y'_i$, is interpreted as the probability that the input $x$ is associated with class $i$.

### 1.2 Cross Entropy Loss Function

The loss function used for multinomial regression is known as the **Cross Entropy Loss** function and is defined with the same motivation as binary cross entropy loss. Here, we want to maximize the prbability that a given input corresponds to a given class $i$ which is the same as minimizing the negative log of the probability. The loss function for a single example $x$ is the sum of the logs of the $k$ output classes:

$$ J(y') = -\sum_{i=1}^{k} 1\{y=i\}\ log\ [\ p(y\ =\ i|x)\ ]$$

$$ J(y') = -\sum_{i=1}^{k} 1\{y=i\}\ log\ [ \frac{e^{z_i}}{\sum_{j=1}^{k} e^{z_j}}\ ]$$

$$ J(y') = -\sum_{i=1}^{k} 1\{y=i\}\ log\ [ y'_i ]$$



In the equation above, we make use of the indicator function $1\{\}$ which evaluates to 1 if the condition in the brackets is true and to 0 otherwise. So, the total loss is a summation across each of the output neurons. An easy way to get an intuition for why this makes sense is to consider two cases. First, consider the case where the neuron with the highest output is associated with the ground truth label for the input $x$. In that case, the total loss would be the negative log of a high probability number. For example, if the probability was 0.9 then the total loss for this case would be $-log(0.9) = .105$. Now consider a case where the correct class had a predicted output probability of .01. In that case, the total loss would be the negative log of a low probability number, $-log(0.01) = 4.61$.

## 2 Data Loading and Preparation

As discussed earlier, we will use the MNIST data for our experiment. It contains `60000` training and `10000` testing grayscale `28x28` images from `10` classes:

The MNIST dataset already comes bundled with PyTorch. PyTorch provides easy access to some standard datasets using `torch.dataset`. You can access all the available datasets in Torchvision for image classification [here](https://pytorch.org/vision/main/datasets.html#image-classification).

### 2.1 Download and Normalize Data

1. We load the training and validation data separately.
2. We specify that the data should be downloaded if it is not present on the system.
3. The data is transformed to PyTorch tensors in the range `[0, 1]` and then **normalized** with the appropriate mean and standard deviation.

In other words, following equation is used for Normalization across each image channel, $C$:

$$ \hat x_C = \frac{x_C - \mu_C}{\sigma_C}$$

For MNIST, we only have have a single channel. Its **mean** and **standard deviation** have already been computed, and turns out to be `(.1307,)` and `(.3081,)` respectively.

#### DataLoader

PyTorch provides a very useful class called `DataLoader` that helps feed the data during the training process. It is primarily used for two purposes.

1. Load a mini-batch of data from a dataset.
2. Shuffle the data (if required).

**What batch size to use?**

We are using a batch size of 32. When you are using a GPU, the maximum batch size is dictated by the memory on the GPU. However, even without the GPU memory limitation, batch size of 32 or smaller is preferred in many applications. See this [funny tweet](https://twitter.com/ylecun/status/989610208497360896?lang=en).

**Why shuffle training set?**

Notice in the code below, we shuffle the training data. This is because the original dataset may have some ordering (e.g. all examples of 0s come first, and then all 1s etc.). This kind of correlation is bad for the training process because the loss calculated over a mini-batch is used to update the weights or network parameters. On the other hand, it makes no sense to shuffle the validation set because validation loss is calculated over the entire validation set.  

## 3 Model Architecture

### 3.1 Single Layer, Multiple Output Architecture

The network architecture shown below is similar to the previous architecture for binary classification but with some important differences. The key differences are summarized below:

1. The image input data is pre-processed in a way that we have not yet discussed (more on this below).
2. We now have 10 neurons to represent the ten different classes (digits: 0 to 9), instead of a single neuron as with binary classification.
3. The activation function is a **softmax** activation rather than a sigmoid activation.
4. The loss function is now **sparse categorical cross entropy**.

Although the diagram looks quite a bit different from previous (single neuron) architectures, it is fundamentally very similar in terms of the processing that takes place during training and prediction.


<img src='https://opencv.org/wp-content/uploads/2023/05/c3_week3_MNIST_network.png' width=900 align='center'><br/>

### 3.2 Fully Connected (Dense) Layers

The neural network architectures we have covered thus far in the course have used "fully connected" layers which are also referred to as "dense" or "linear" layers. This is very common, but as the number of inputs and neurons in each layer becomes larger, the number of trainable parameters grows exponentially. The figure below shows two examples of fully connected layers. When depicting neural network architectures with fully connected layers, the connections are typically omitted with the understanding that 'dense' or 'fully connected' is assumed.

<img src='https://learnopencv.com/wp-content/uploads/2022/01/c4_02_dense_layers.png' width=700 align='center'>

As we will see later in the course, when working with images, the number of parameters can become exceedingly large as the number of neurons and the number of layers in the network is increased. For example, it is not uncommon for state-of-the-art networks to contain millions of parameters. Larger networks hold the potential to exceed the performance of smaller networks, but that comes at the cost of much longer training times. In order to mitigate these issues, we will see that the data in the network is sometimes down-sampled at intermediate layers, which reduces the number of parameters. One approach that is used to down-sample data in CNNs is called 'pooling.' Another approach called "dropout", is a stochastic regularization technique that is used to reduce overfitting by randomly dropping a percentage of neurons from the network (along with their associated connections) which also reduces the number of trainable parameters in the network. We will cover these topics in more detail later in the course.

## 4 Training

### 4.1 Define Train Step

This is the Training routine which does the following:

1. It takes batches of data from train dataloader
2. The training data is passed through the network
3. Compute the Cross Entropy loss using the predicted output and the training labels
4. To avoid gradient accumulation, remove previous gradients using optimizer.zero_grad
5. Compute Gradients using the backward function
6. Update the weights using the optimizer.step function and repeat until all the data is passed through the network.

**Note** : During training we will use the [nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss) that combines `nn.LogSoftMax` (Log of SoftMax) and `nn.NLLLoss` (Negative Log Likelihood Loss). This also means that when we do inference, we have to use `softmax` on the raw output to convert it to probabilities.
