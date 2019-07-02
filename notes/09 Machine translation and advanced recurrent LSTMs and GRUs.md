## Machine translation and advanced recurrent LSTMs and GRUs

### 机器翻译

机器翻译的目的：
$$\hat{e}=\text{argmax}_ep(e|f)=\text{argmax}_ep(f|e)p(e)$$  
使用贝叶斯公式，翻译模型$p(f|e)$在平行语料上训练得到，语言模型$p(e)$在未对齐的原文语料上训练。

传统方法：对齐-解码  
RNN方法：直接预测“下一个”单词

### RNN方法发展  

1. Encoder和Decoder训练不同的权值矩阵
2. Decoder中的隐藏层的输入来自3个方面——上一个时刻的隐藏层，隐藏层的最终输出，前一个预测结果，可以防止模型重复生成同一个单词
3. 使用深度RNN
4. 使用双向Encoder
5. 逆转原文词序，防止梯度消失

### 更好的单元

GRU/LSTM

### 最新改进

softmax的问题：无法辨认出新词
解决方案：使用指针来解决问题，返回的是指向前x个单词的指针，训练这个x