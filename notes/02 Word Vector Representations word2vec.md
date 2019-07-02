## Word Vector Representation & word2vec

### How to represent the meaning of a word?

Wordnet:  
taxonomy(分类学) information about words.


	from nltk import wordnet as wn
	panda = wn.synset('panda.n.01')
	hyper = lambda s: s.hypernyms()
	list(panda.closure(hyper))

Use the nltk to get a hold of word net.  

### Problem with the discrete representation

1. Synonyms:  
`adept, expert, good, practiced, proficient, skillful`  
2. Missing new words  
3. Subjective  
4. Requires human labor to create and adapt  
5. Hard to compute accurate word similarity  

Usually regards words as atomic symbols.  
We call this a one-hot representation.  
Our query and document vectors are orthogonal(正交的).  

### Distributional similarity based representations

“You shall know a word by the company it keeps”  (J. R. Firth 1957: 11)  
build a dense vector for each word type, chosen so that it is good at predicting other words appearing in its context.  

## Word2Vec

### Basic idea
predict between a center word wt and context words  
p(context|wt) = ...  
has a loss funtion:  
$$ J = 1 − p(w−t |wt)  $$  

### Main idea
Two algorithm:  

1. Skip-Gram(SG)  
Predict context words given target (position independent)
2. Continuous Bag of Word(CBOW)  
Predict target word from bag-of-words context

Two training methods:

1. Hierarchical softmax  

2. Negative sampling  

###Details of SG

Loss function:  
$$ J'(\theta) = \prod\_{t = 1}^T \prod\_{-m \leq j \leq m, j \neq 0} p(w\_{t+j}|w_t\theta)$$  

Tweak that(negative log likelihood):  
$$ J(\theta) = -\cfrac{1}{T} \sum\_{t = 1}^T \sum\_{-m \leq j \leq m, j \neq 0} p(w\_{t+j}|w_t)$$  

How to calculate $ p(o|c) $:

$$ p(o|c) = \cfrac{exp(u_o^Tv_c)}{\sum\_{w=1}^vexp(u_w^Tv_c)} $$

o is the output(outside) word index, c is the center word index, v & u are "center" and "outside" vectors of indices c and o.  

Softmax is a standard way to turn numbers into a probability distribution.   

###Train the model

$$
\theta = 
\begin{bmatrix}  
   v\_{a}      \\\\  
\vdots      \\\\  
v\_{zebra}  \\\\  
   u\_{a}      \\\\  
\vdots      \\\\  
u\_{zebra}  
\end{bmatrix}  
$$

Optimize these parameters.

Note: Every word has two vectors.  

###Highlight - Sentence Embedding

We can compute sentence similarity using the inner product.  
Use as features for sentence classification(e.g. sentiment analysis).  

1. Bag-of-words(BoW): use words' embedding vectors' average.  
2. Recurrent neural network, recursive neural network, convolutional neural network.

Paper: A Simple but Tough-to-beat Baseline for Sentence Embeddings  

weighted Bag-of-words + remove some special direction  

Step 1:  
$$ v\_s\leftarrow\cfrac{1}{\vert s \vert} \sum\_{w \in s}\cfrac{a}{a+p(w)} $$  
a is a constant, p(w) is the frequency of this word.  

Step 2:  
Computer the first principle component $u$ of $v\_s$  
$$ v\_s \leftarrow v\_s - u \cdot u^T \cdot v\_s $$  

###Use gradient to optimise the model

Focus on word representation as center word.  

$$  
\begin{align}  
\cfrac{\partial log(p(o|c))}{\partial v_c}  
& = \cfrac{\partial}{\partial v_c} log \cfrac{exp(u_o^Tv_c)}{\sum\_{w=1}^v exp(u_w^Tv_c)} \\\\  
& = \cfrac{\partial}{\partial v_c} log exp(u_o^Tv_c) - \cfrac{\partial}{\partial v_c} log \sum\_{w=1}^v exp(u_w^Tv_c) \\\\  
& = u_o - \cfrac{1}{\sum\_{w=1}^{v} exp(u_w^Tv_c)} \cdot \cfrac{\partial}{\partial v_c} \sum\_{x=1}^v exp(u_x^Tv_c) \\\\  
& = u_o - \cfrac{1}{\sum\_{w=1}^{v} exp(u_w^Tv_c)} \cdot (\sum\_{x=1}^v exp(u_x^Tv_c)u_x) \\\\  
& = u_o - \sum\_{x=1}^v \cfrac{exp(u_x^Tv_c)}{\sum\_{w=1}^v exp(u_w^Tv_c)} u_x \\\\  
& = u_o - \sum\_{x=1}^v p(x|c) u_x \\\\  
\end{align}  
$$  

$u_o$ is the actual output context word appeared, $\sum\_{x=1}^v p(x|c) u_x$ has the form of an exception.

Use Stochastic Gradient Descent instead of Basic Gradient Descent.  

