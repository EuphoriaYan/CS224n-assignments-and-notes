# CS224N: Assignment #3

## 1 A window into NER (30 points)

### (a) (5 points)

(i) 南京市长江大桥  
(ii) 一般来说命名实体频率都很低  
(iii) 词性，词语形态

### (b) (5 points) 

(i) e(t) is 1 * (2w + 1)D. W is (2w + 1)D * H. U is H * C.
(ii) O((2w + 1)DHT)

### (c) (15 points) 

q1_window.py

### (d) (5 points) 

(i)  

	DEBUG:Token-level confusion matrix:
	go\gu   PER     ORG     LOC     MISC    O    
	PER     2968    26      84      16      55   
	ORG     147     1621    131     65      128  
	LOC     48      88      1896    26      36   
	MISC    37      40      54      1030    107  
	O       42      46      18      39      42614

## 2 Recurrent neural nets for NER (40 points)

### (a) (6 points)

### (b) (2 points)

### (c) (5 points)

q2_rnn_cell.py

### (d) (8 points)

q2_rnn.py

### (e) (12 points)

q2_rnn.py

### (f) (3 points)

q2_rnn.py

### (g) (6 points)

\# TODO

## 3 Grooving with GRUs (30 points)

### (a) (4 points)


### (b) (6 points)


### (c) (6 points)

q3_gru_cell.py

### (d) (6 points)

q3_gru.py

### (e) (5 points)

### (f) (3 points)