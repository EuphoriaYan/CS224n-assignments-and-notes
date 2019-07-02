## Neural Machine Translation and Models with Attention

### 机器翻译

同时涉及到语言分析与理解，传统的NLU测试  

Neural machine translation: 用一个大型神经网络来做整个翻译过程的系统。  
架构：Encoder+Decoder  
Encoder最后一层的输出是原文的总结；

### Attention

朴素encoder-decoder的问题是，只能用固定维度的最后一刻的encoder隐藏层来表示源语言Y，必须将此状态一直传递下去。
Attention：隐式对齐，将encoder的历史状态视作随机读取内存，增加了记忆的持续时间。
Doubly attention：同时注意Encoder和Decoder  
用旧模型的语言学思想拓展attention  

### Decoder

找出最可能的译文：  
朴素思路：全部生成，打分；  
只根据历史词；  
贪心搜；  
Beam Search；


