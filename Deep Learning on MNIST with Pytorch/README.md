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




