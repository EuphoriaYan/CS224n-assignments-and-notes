## Gated recurrent units and further topics in NMT

### GRU和LSTM

总体上来说，LSTM的记忆可以比RNN持续更长的step（大约100）

训练技巧：  

- 将递归权值矩阵初始化为正交
- 将其他矩阵初始化为较小的值
- 将forget gate偏置设为1：默认为不遗忘
- 使用自适应的学习率算法：Adam、AdaDelta
- 裁剪梯度的长度为小于1-5
- 在Cell中垂直应用Dropout而不是水平Dropout
- 通常需要训练很长时间

### BLEU

IBM研究出了一种简单有效的翻译评价策略——BLEU，比较标准译文与机翻译文中N-Gram的重叠比率（0到1之间）来衡量机翻质量

### 大词表

早期——小词表 ，hierarchical softmax，  
后期——Large-vocab NMT  
训练时每次只在词表的一个小子集上训练，用词相似的文章进入同一个子集。


