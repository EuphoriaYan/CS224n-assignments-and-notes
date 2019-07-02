## Recurrent Neural Networks and Language Models

### 语言模型
语言模型是计算一个单词序列的概率的模型。

$$  
\begin{equation}  
P(w_1,…,w_m) = \prod\_{i=1}^{i=m} P(w_i | w_1, …, w\_{i-1}) \approx \prod\_{i=1}^{i=m} P(w_i | w\_{i-n}, …, w\_{i-1})  
\label{eqn:nat_model}  
\end{equation}  
$$  

常用极大似然估计来估计这个概率。

$$  
\begin{align} 
p(w_2 | w_1) &= \dfrac {count(w_1,w_2)}{count(w_1)} \\\\  
p(w_3 | w_1, w_2) &= \dfrac {count(w_1,w_2,w_3)}{count(w_1, w_2)}  
\end{align}  
$$

在数据量足够的情况下，n-gram中的n越大，一般模型效果越好。  
实际情况中往往需要平滑方法，需要考虑内存与性能的权衡。  

### RNN方法

新的语言模型是利用RNN对序列建模，类似于seq2seq的decoder。

$$
h_t = \sigma(W^{(hh)} h\_{t-1} + W^{(hx)} x_t)
$$

### 梯度消失/梯度爆炸

在前向传播的时候，前面的$x_t$反复乘上W，导致对后面的影响很小。 

$$
\begin{equation} 
\dfrac{\partial E}{\partial W} = \sum\_{t=1}^{T}\sum\_{k=1}^{t} \dfrac{\partial E_t}{\partial y_t} \dfrac{\partial y_t}{\partial h_t} (\prod\_{j=k+1}^{t}\dfrac{\partial h_j}{\partial h_{j-1}}) \dfrac{\partial h_k}{\partial W}  
\end{equation}
$$

记$\beta_W$ 和 $\beta_h$分别为矩阵和向量的范数(L2)，

$$
\begin {equation} 
\parallel \dfrac{\partial h_t}{\partial h_k} \parallel = \parallel \prod\_{j=k+1}^t \dfrac{\partial h_j}{\partial h\_{j-1}}\parallel \leq (\beta_W \beta_h)^{t-k} 
\label{eqn:bp_rnn_k_norm_total} 
\end {equation}
$$

在$\beta_W \beta_h$大于1时，浮点数运算会产生溢出（NaN），一般可以很快发现。这叫做梯度爆炸。小于1，或者下溢出并不产生异常，难以发现，但会显著降低模型对较远单词的记忆效果，这叫做梯度消失。

### 解决梯度爆炸/梯度消失

防止梯度爆炸：  
当梯度的长度大于某个阈值的时候，将其缩放到某个阈值。无法推广到梯度消失，无法规定最低阈值。

减缓梯度消失：
初始化为单位阵；

### RNN语言模型的应用

NER，Semantic Analysis，机器翻译等等。

深度双向RNN的层数不是越多越好。