# CS224N: Assignment #2

## 1 Tensorflow Softmax (25 points)

### (a) (5 points)

q1_softmax.py

### (b) (5 points) 

q1_softmax.py

### (c) (5 points) 

q1_classifier.py

### (d) (5 points) 

q1_classifier.py

### (e) (5 points) 

q1_classifier.py

## 2 Neural Transition-Based Dependency Parsing (50 points)

### (a) (6 points)
| stack        | buffer          | new dependency  | transition     |
|:---------- |:-------------|:--------------------|:-------------|
| [ROOT]    | [I, parsed, this, sentence, correctly] |  | Initial Configuration |
| [ROOT, I]     | [parsed, this, sentence, correctly]      |  | SHIFT |
| [ROOT, I, parsed]     | [this, sentence, correctly]      |  | SHIFT |
| [ROOT, parsed]     | [this, sentence, correctly]      | parsed-> I | LEFT-ARC |
| [ROOT, parsed, this]     | [sentence, correctly]      |  | SHIFT |
| [ROOT, parsed, this, sentence]     | [correctly]      |  | SHIFT |
| [ROOT, parsed, sentence]     | [correctly]      | sentence->this | LEFT-ARC |
| [ROOT, parsed]     | [correctly]      | parsed->sentence | RIGHT-ARC |
| [ROOT, parsed, correctly]     | []      |  | SHIFT |
| [ROOT, parsed]     | []      | parsed->correctly | RIGHT-ARC |
| [ROOT]     | []      | ROOT->parsed | RIGHT-ARC |

### (b) (2 points)

每个词只会被移进和归约各一次，因此n个单词需要2n次操作；

### (c) (6 points)

q2_parser_transitions.py

### (d) (6 points)

q2_parser_transitions.py

### (e) (4 points)

q2_initialization.py

### (f) (2 points)

$$
E\_{p\_{drop}} = p\_{drop}(0) + (1-p\_{drop})\gamma h_i = h_i
$$

So, $\gamma = \cfrac{1}{1-p\_{drop}}$

### (g) (4 points)

(i) 减少震荡  
(ii) 加速训练  

### (h) (20 points)

	924/924 [==============================] - 44s - train loss: 0.0602    
	Evaluating on dev set - dev UAS: 88.44
	New best dev UAS! Saving model in ./data/weights/parser.weights
	================================================================================
	TESTING
	================================================================================
	Restoring the best model weights found on the dev set
	Final evaluation on test set - test UAS: 88.69
	Writing predictions
	Done!

## 3 Recurrent Neural Networks: Language Modeling (25 points)

### (a) (5 points)

$$CE(y^{(t)},\hat{y}^{t})=-\log \hat{y}_i^{(t)}=\log \frac{1}{\hat{y}^{t}}$$
$$PP^{(t)}(y^{(t)},\hat{y}^{(t)})=\frac{1}{\hat{y}^{(t)}}$$
$$CE(y^{(t)},\hat{y}^{t})=\log PP^{(t)}(y^{(t)},\hat{y}^{(t)})$$

### (b) (8 points)

首先设  
$$  
f_1^{(t)} = h^{(t-1)}H + e^{(t)}I + b_1 \\\\   
f_2^{(t)} = h^{(t)}U + b_2  
$$  
计算残差：  
$$  
\begin{align}  
\delta_2^{(t)} &= \cfrac{\partial J^{(t)}}{\partial f_2^{(t)}}  
= \hat{y}^{(t)} - y^{(t)} \\\\  
\delta_1^{(t)} &= \cfrac{\partial J^{(t)}}{\partial f_1^{(t)}} \\\\  
&=\cfrac{\partial J^{(t)}}{\partial f_2^{(t)}} \cdot \cfrac{\partial f_2^{(t)}}{\partial h^{(t)}} \cdot \cfrac{\partial h^{(t)}}{\partial f_1^{(t)}} \\\\  
&= \delta_2^{(t)} \cdot \cfrac{\partial f_2^{(t)}}{\partial h^{(t)}} \cdot \cfrac{\partial h^{(t)}}{\partial f_1^{(t)}} \\\\  
&= \delta_2^{(t)} \cdot U^T \circ h^{(t)} \circ (1 - h^{(t)})   
\end{align}  
$$  
最终结果：

$$  
\begin{align}  
\cfrac{\partial J^{(t)}}{\partial b_2} &= \cfrac{\partial J^{(t)}}{\partial f_2^{(t)}} \cfrac{\partial f_2^{(t)}}{\partial b_2} \\\\  
&= \delta_2^{(t)} = \hat{y}^{(t)} - y^{(t)}  
\end{align}
$$

$$  
\begin{align}  
\cfrac{\partial J^{(t)}}{\partial L\_{x^{(t)}}} &= \cfrac{\partial J^{(t)}}{\partial f_1^{(t)}} \frac{\partial f_1^{(t)}}{\partial e^{(t)}} \cfrac{\partial e^{(t)}}{\partial L\_{x^{(t)}}}  \\\\  
&= \delta_1^{(t)} I^T   
\end{align}  
$$

$$
\begin{align} 
\cfrac{\partial J^{(t)}}{\partial I} &= % 
\cfrac{\partial J^{(t)}}{\partial f_1^{(t)}} \cfrac{\partial f_1^{(t)}}{\partial I}  \\\\  
&=  (e^{(t)})^{T} \delta_1^{(t)}  
\end{align}  
$$

$$
\begin{align} 
\cfrac{\partial J^{(t)}}{\partial H} &= % 
\cfrac{\partial J^{(t)}}{\partial f_1^{(t)}} \cfrac{\partial f_1^{(t)}}{\partial H} \\\\  
&= (h^{(t-1)})^T \delta_1^{(t)}  
\end{align}  
$$

$$
\begin{align} 
\cfrac{\partial J^{(t)}}{\partial h^{(t-1)}} &= % 
\cfrac{\partial J^{(t)}}{\partial f_1^{(t)}} \cfrac{\partial f_1^{(t)}}{\partial h^{(t-1)}} \\\\  
&= \delta_1^{(t)} H^T   
\end{align}  
$$

### (c) (8 points)

\# TODO

### (d) (4 points)

Forward:  
$$ O\left( (d \times D_h) + D_h^2 + (D_h \times |V|) \right) $$  

Backward:  
$$ O\left( \tau \left((d \times D_h) + D_h^2 +  D_h \times |V| \right)\right) $$  
