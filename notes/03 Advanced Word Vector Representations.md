## Advanced Word Vector Representations

### Stochastic gradients with word vectors

In each window, we only have at most 2m + 1 words.

$$  
\nabla\_{\theta}J\_{t}(\theta) =  
\begin{bmatrix}  
0  \\\\  
\vdots      \\\\  
\nabla\_{v\_{like}}  \\\\  
0  \\\\  
\nabla\_{u\_{I}}  \\\\  
\vdots      \\\\  
\nabla\_{u\_{learning}}  \\\\  
\vdots  
\end{bmatrix}
$$

We may only update the word vectors that actually appear.  
We can use sparse matrix or hash to implement it.  
Most of the objective functions in this class are not convex, so initialization does matter, but if we initialize with small random numbers in these vectors, it does not tend to be a problem.  

### Use negative sampling  

$$ p(o|c) = \cfrac{exp(u_o^Tv_c)}{\sum\_{w=1}^vexp(u_w^Tv_c)} $$

The upper part is pretty simple, but the lower part should do gigantic sum, and this sum goes over the entire vocabulary.  
The main idea behind skip-gram is a very neat trick, which is we'll just train a couple of binary logistic regressions for the true pair.  
We'll just take a couple of random words and say how about these random words from the rest of the corpus don't co-occur.  

Word2Vec: Overall objective function:  
$$ J(\theta) = \cfrac{1}{T}\sum\_{t=1}^T J_t(\theta) $$  
$$ J_t(\theta) = log\sigma(u_o^Tv_c) + \sum\_{i=1}^T \mathbb{E}\_{j\sim P(w)}[log\sigma(-u_j^Tv_c)] $$

T here corresponds to each window as you go through the corpus.  
Two terms in $J_t(\theta)$, the first one is just a log probability of these two center words and outside words co-occuring.  
Randomly subsample a couple of words from the corpus, and for each of these, we will essentially try to minimize their probability of co-occurring.  
We can do this instead of going through all the different ones saying which word doesn't appear.  

1. We take k(about 5~10) negative samples.
2. Maximize probability that real outside word appears, minimize probability that random words appear around center word.  
$$ P(w) = U(w)^{3/4}/Z $$  
We sample them from a simple uniform or unigram distribution.The unigram distribution U(w) raised to the 3/4 power so less frequent words be sampled more often.  
The P(w) is an empirical formula, a very simple thing.  

### Assignment 1: The CBOW

Main idea for CBOW: Predict center word from sum of surrounding word vectors.  
Take the derivatives of CBOW function.  

### Word2vec put similar words nearby in space

"Cluster around similar kinds of meaning"  
We can use PCA（主成分分析） visualization of these word vectors.  

### SVD Based methods

Skip-Gram or CBOW capture cooccurrence of words one at time.  
Another choose: capture cooccurrence counts directily.  
A method that came historically before word2vec:  
With a co-occurrence matrix X, we can:  

1. Capture both syntactic(POS) and semantic information.  
2. Word-document co-occurrence matrix will give general topics leading to "Latent Semantic Analysis"  .  

Let our corpus contain just three sentences and the window size be 1:

1.  I enjoy flying.  
2.  I like NLP.  
3.  I like deep learning.  

The resulting counts matrix will then be:   
$$  
\begin{array}{lc}  
\mbox{}&  
\begin{array}{cc}
I & like & enjoy & deep & learning & NLP & flying & .  
\end{array} \\\\  
\begin{array}{c}  
I \\\\  
like \\\\  
enjoy \\\\  
deep \\\\  
learning \\\\  
NLP \\\\  
flying \\\\  
.  
\end{array}&  
\left[\begin{array}{cc}  
0 & 2 & 1 & 0 & 0 & 0 & 0 & 0\\\\  
2 & 0 & 0 & 1 & 0 & 1 & 0 & 0\\\\  
1 & 0 & 0 & 0 & 0 & 0 & 1 & 0\\\\  
0 & 1 & 0 & 0 & 1 & 0 & 0 & 0\\\\  
0 & 0 & 0 & 1 & 0 & 0 & 0 & 1\\\\  
0 & 1 & 0 & 0 & 0 & 0 & 0 & 1\\\\    
0 & 0 & 1 & 0 & 0 & 0 & 0 & 1\\\\  
0 & 0 & 0 & 0 & 1 & 1 & 1 & 0  
\end{array}\right]  
\end{array}  
$$

It's not a very ideal word vector -- new words, high-dimension.  
Solution: Low dimensional vectors, usually 25-1000 dimensions, similar to word2vec.  
Method1:  
Apply SVD to the cooccurrence matrix.  
Problem: (the, he, has) are too frequent. -- Ignore or cap it(min(X,t),with t~100).  
Problem with SVD: Cimputational cost.  o(mn^2)  


### Count based vs direct prediction

LSA, HAL  
COALS, Hellinger-PCA

Advantages: Fast training, efficient usage of statistics.  
Disadvantages: Primarily used to capture word similarity, disproportionate（不成比例的） importance given to large counts.

Skip-gram/CBOW  
NNLM, HLBL, RNN  

Disadvantages: Scales with corpus size, inefficient usage of statistics.
Advantages: Generate improved performance on other tasks, can capture complex patterns
beyond word similarity.  

### GloVe: Combining the best of two world

GloVe :Global Vectors model  

Loss function:  
$$ J(\theta) = \cfrac{1}{2} \sum\_{i,j=1}^W f(P\_{ij})(u_i^T v_j - log P\_{ij})^2 $$  
Co-occurrence matrix:  P  
Minimize the distance between the inner product $u_i^T v_j$, and the log count of these two words co-occurrence.  
The function $f(P\_{ij})$ can lower frequent co-occurrences.  

- Fast training  
- Scalable to huge corpora  
- Good performance, even with small corpus  

### Two sets of vectors

The best solution is to simply sum them up:  
$$ X\_{final} = U+V $$  

### Highlight: Polysemy

Word vectors can capture polysemy.  
Word vectors are linear superposition（线性叠加） of each sense vector.  
Sense/context vectors can be recovered by sparse coding.  

