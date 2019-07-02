## End-to-end models for Speech Processing

### Automatic Speech Recognition（ASR）

传统ASR  
传统做法的主体是生成式语言模型，建模声学信号与文本的发音特征的联合概率，但pipeline的不同部分掺杂了不同的机器学习模型；

近现代ASR  
神经网络兴起之后，人们发现传统pipeline中的每个模型都可以被一种对应的神经网络所替代，并且取得更好的效果；

动机——用一个统一的大模型来取代这些小神经网络

end-to-end ASR，直接从音频到字符

### Connectionist Temporal Classification

主体是Bi-RNN+Softmax

问题是拼写不正确，Google集成语言模型，修正了问题。

### 带Attention的序列到序列训练

将音频视作sequence，文本视作另一个sequence。  
Listen Attend and Spell， 树形Encoder，很强大；  
限制：必须等到用户说完话之后才能开始识别，attention是计算瓶颈，输入的长度对准确率影响特别大； 

### 在线seq2seq模型

Neural Transducer：根据一个定长的输入序列片段产生输出，不要要前一个输出，依然需要空白符，依然需要alignment（哪些字母属于一个词）

采用了柱搜索找出最可能的路径（分词）；





