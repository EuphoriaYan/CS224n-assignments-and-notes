# CS224N: Assignment #1

## 1 Softmax (10 points)

### (a) (5 points)

*Proof.*  $\forall 1 \le i \le dim(x)$  

$$
(softmax(x + c))_i  
=\cfrac{exp(x_i+c)}{\sum\_{j=1}^{dim(x)}exp(x_j+c)}  
=\cfrac{exp(c)exp(x_i)}{exp(c)\sum\_{j=1}^{dim(x)}exp(x_j)}  
=\cfrac{exp(x_i)}{\sum\_{j=1}^{dim(x)}exp(x_j)}  
=(softmax(x))_i  
$$

### (b) (5 points) 

q1_softmax.py

## 2 Neural Network Basics (30 points)

### (a) (3 points)

$$\sigma'(x)=\sigma(x)(1-\sigma(x))$$

$$
\begin{align}  
\sigma(x) &= \cfrac{1}{1 + e^{-x}}  \\\\   
                &= \frac{e^{x}}{1 + e^{x}} \\\\ 
\end{align}
$$
$$
\begin{align} 
\cfrac{\partial}{\partial x} \sigma(x) &= \cfrac{e^{x}  (1 + e^{x}) – (e^{x} e^{x})}{(1 + e^{x})^2}  \\\\   
&= \cfrac{e^x}{1 + e^x} \cdot \cfrac{1}{1 + e^x} \\\\   
&= \sigma(x)  \sigma(-x) \\\\  
&= \sigma(x) (1 – \sigma(x))
\end{align}
 $$


### (b) (3 points)

$$\cfrac{\partial CE(y, \hat{y})}{\partial \theta}=\hat{y}-y $$  

Let:  
$$  
\begin{align} 
  f_i &= e^{\theta_i} \\\\  
  g_i &= \sum\_{k=1}^K e^{\theta_k} \\\\  
  S_i &= \frac{f_i}{g_i} \\\\  
  \cfrac{\partial S_i}{\partial \theta_j} &= \cfrac{f'_i g_i – g'_i f_i}{g_i^2}  
\end{align}
$$  

When i == j:  
$$  
f'_i = f_i \\\\
g'_i = e^{\theta_j} \\\\  
\begin{align} 
\cfrac{\partial S_i}{\partial \theta_j} &= \cfrac{e^{\theta_i} \sum_k e^{\theta_k} – e^{\theta_j} e^{\theta_i} }{ (\sum_k e^{\theta_k})^2 } \\\\  
&= \cfrac{e^{\theta_i}}{\sum_k e^{\theta_k}} \cdot \cfrac{\sum_k e^{\theta_k} – e^{\theta_j}}{\sum_k e^{\theta_k}} \\\\  
&= S_i (1 – S_i)
\end{align}
$$  

When i!=j:  
$$  
f'_i = 0 \\\\  
g'_i = e^{\theta_j} \\\\  
\begin{align}  
\cfrac{\partial S_i}{\partial \theta_j}  &= \cfrac{0 – e^{\theta_j} e^{\theta_i}}{(\sum_k e^{\theta_k})^2} \\\\  
&= – \cfrac{e^{\theta_j}}{\sum_k e^{\theta_k}} \cdot \cfrac{e^{\theta_i}}{\sum_k e^{\theta_k}} \\\\   
&= -S_j S_i  
\end{align}  
$$  

$$  
\begin{align} 
\cfrac{\partial CE}{\partial \theta_i}
&= – \sum_k y_k \cfrac{\partial log S_k}{\partial \theta_i} \\\\  
&= – \sum_k y_k \cfrac{1}{S_k} \cfrac{\partial S_k}{\partial \theta_i} \\\\  
&= – y_i (1 – S_i) – \sum\_{k \ne i} y_k \cfrac{1}{S_k} (-S_k S_i) \\\\  
&= – y_i (1 – S_i) + \sum\_{k \ne i} y_k S_i \\\\  
&= – y_i + y_i S_i + \sum\_{k \ne i} y_k S_i \\\\  
&= S_i(\sum_k y_k) – y_i   
\end{align}
$$  

Becasue $\sum_k y_k = 1$, so $\cfrac{\partial CE}{\partial \theta_i} = \hat{y}-y$


### (c) (6 points)

Let $z_2=hW_2+b_2$, and $z_1=xW_1+b_1$, then  

$$  
\begin{equation}  
\begin{split}  
\delta_1&=\cfrac{\partial CE}{\partial z_2}=\hat{y}-y \\\\  
\delta_2&=\cfrac{\partial CE}{\partial h}=\delta_1 \cfrac{\partial z_2}{\partial h}=\delta_1 W_2^T \\\\  
\delta_3&=\cfrac{\partial CE}{\partial z_1}=\delta_2 \cfrac{\partial h}{\partial z_1}=\delta_2 \sigma'(z_1) \\\\  
\cfrac{\partial CE}{\partial x}&=\delta_3 \cfrac{\partial z_1}{\partial x}=\delta_3 W_1^T  
\end{split}  
\end{equation}  
$$

### (d) (2 points)

The count of parameters is $D_x \cdot H + H + D_y \cdot H + D_y$.

### (e) (4 points)

q2_sigmoid.py

### (f) (4 points)

q2_gradcheck.py

### (g) (8 points)

q2_neural.py

## 3 word2vec (40 points + 2 bonus)

### (a) (3 points)

Let $\hat{y}$ be the column vector of the softmax prediction of words, and $y$ be the one-hot label which is also a column vector. Then
$$
\frac{\partial J}{\partial v_c}=U(\hat{y}-y).
$$

### (b) (3 points)

$$
\frac{\partial J}{\partial U}=v_c(\hat{y}-y)^T.
$$

### (c) (6 points)

$$
\begin{equation}
\begin{split}
\frac{\partial J}{\partial v_c} &= (\sigma(u_o^Tv_c)-1)u_o-\sum_{k=1}^K (\sigma(-u_k^Tv_c)-1)u_k \\\\  
\frac{\partial J}{\partial u_o} &= (\sigma(u_o^Tv_c)-1)v_c \\\\  
\frac{\partial J}{\partial u_k} &= -(\sigma(-u_k^Tv_c)-1)v_c, \text{ for all }k=1,2,...,K
\end{split}
\end{equation}
$$

### (d) (8 points)

For SG:  
$$
J\_{skip-gram}(word\_{c-m \dots c+m}) = \sum\limits\_{-m \leq j \leq m, j \ne 0} F(w\_{c+j}, v_c)  
$$


CBOW:  
$$  
\hat{v} = \sum\limits\_{-m \leq j \leq m, j \ne 0} v\_{c+j}  
$$  

$$  
J\_{CBOW}(word\_{c-m \dots c+m}) = F(w_c, \hat{v})  
$$  

### (e) (12 points)

q3_word2vec.py

### (f) (4 points)

q3_sgd.py

### (g) (4 points)

q3_run.py

**Solution:**

![](assignment1/q3_word_vectors.png)

### (h) (Extra: 2 points)

q3_word2vec.py

## 4 Sentiment Analysis (20 points)

### (a) (2 points)

q4_sentiment.py

### (b) (1 points)

**Solution:**

To avoid overfitting to the training examples.

### (c) (2 points)

q4_sentiment.py

**Solution:**

	bestResult = max(results, key=lambda x: x["dev"])

###(d) (3 points)

Some possible reasons:

* Higher dimensional word vectors may contain more information
* GloVe vectors were trained on a larger corpus
* GloVe take the use of global vector, but word2vec not.

### (e) (4 points)

**Solution:**

![](assignment1/q4_reg_v_acc.png)

### (f) (4 points)

**Solution:**

![](assignment1/q4_dev_conf.png)

### (g) (4 points)

**Solution:**

sarcasm（讽刺）: nothing is sacred in this gut-buster.  
