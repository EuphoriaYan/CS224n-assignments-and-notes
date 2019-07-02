## Word Window Classification and Neural Networks

### Classification intuition

Training data: $\\\{ x_i,y_i \\\} \_{i=1}^N$
$x_i$ is a d-dimentional vector, $y_i$ is a C-dimentional one-hot vector.  
Use logistic regression & SVM  

### The softmax

Logistic regression = Softmax classification  
$$ p(y|x) = \cfrac{exp(W_y x)}{\sum\_{c=1}^C exp(W_c x)} $$  
Loss for softmax: Cross entropy  
$$ J(\theta) = \cfrac{1}{N}\sum\_{i=1}^N -log(\cfrac{e^{f_{y_i}}}{\sum\_{c=1}^C e^{f_c}} ) $$  

Regularization:  
$$ J(\theta) = \cfrac{1}{N}\sum\_{i=1}^N -log(\cfrac{e^{f_{y_i}}}{\sum\_{c=1}^C e^{f_c}} ) + \lambda \sum_k \theta_k^2 $$  
Regularization will prevent overfitting when we have a lot of features or a very poweful model.  
It will encourage the model to keep all the weights as small as possible and as close as possible to zero.  

### Classification difference with word vectors

Problem:  

1. However, in word vector training or other deep learning task, the vector will be very large(word embedding matrix).  
We should use regularization to prevent overfitting.  
2. If we train the word vectors, words from pre-training that do NOT appear in training stay.  
For example, In training data: "TV" and "telly", only in testing data: "television"

If you only have a small training data set, don't train the word vectors.  

- The word vector matrix L is also called lookup table  
- Word vectors = word embeddings = word representations (mostly)  
- Mostly from methods like word2vec or Glove  

### Window classification

Train softmax classifier by assigning a label to a center word and concatenating all word vectors surrounding it.  
But softmax only gives linear decision boundaries in the original space.  
Neural networks can learn much more complex functions and nonlinear decision boundaries.  

### Basic idea of neural network

Omitted.  
Use neural network for window classification.  

### Objective function - the max-margin loss

Make the score of the true windows larger than the one of the corrupt windows smaller or lower.  

$$minimize J = \max(s_c - s, 0)$$  

However, this function is too loose.  

$$minimize J = \max(\Delta+s_c - s, 0)$$  

Let's say $\Delta = 1$, it's a hyperparameter.  

$$minimize J = \max(1+s_c - s, 0)$$  

Just like SVM, use geometric method to solve problem.  

### Backpropagation

Omitted.

