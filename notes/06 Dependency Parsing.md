## Dependency Parsing

Two views of linguistic structure:  

1. Constituency = phrase structure grammar = context-free grammars (CFGs)  
$$ S \rightarrow  NP + VP $$  
$$ VP -> V + PP $$  

2. Dependency structure  

Attachment ambiguities:  
Scientists study whales from space.  
"from space" attach to study or whales？  

### The rise of annotated data（标注数据集）: Universal Dependencies treebanks

In the beginning, building a treebank seems a lot slower and less useful than building a grammar.  
But a treebank give us many things:  

- Reusability of the labor  
	- Many parsers, part-of-speech taggers, etc. can be built on it  
	- Valuable resource for linguistics  
- Broad coverage, not just a few intuitions(POS, NER, etc..)  
- Frequencies and distributional information  
- A way to evaluate systems  

The arrow connects a head (governor, superior, regent) with a dependent (modifier, inferior, subordinate).  
Usually, dependencies form a tree.  
Usually add a fake ROOT so every word is a dependent of precisely 1 other node.  

Sources of information for dependency parsing:  

1. Bilexical affinities（双词汇亲和，个人理解是语义和POS均相似）
2. Dependency distance
3. Intervening material（中间词语，如标点）
4. Valency of heads（词语配价，依赖者多少）

We **can't** restore a sentence from a denpendency tree.  

Methods of Dependency Parsing:  

1. Dynamic programming  
2. Graph algorithms  
3. Constraint Satisfaction  
4. “Transition-based parsing” or “deterministic dependency parsing”  

### Highlight: Improving Distributional Similarity with Lessons Learned from Word Embeddings

Count-based distributional models:

- SVD
- PPMI(Positive Pointwise Mutual Information)

Neural network-based models:

- SGNS (Skip-Gram Negative Sampling)/ CBOW
- GloVe

Conventional wisdom:  
Neural-network based models > Count-based models  

Hyperparameters and system design choices more important, not the embedding algorithms themselves.

Summary: 

- This paper challenges the conventional wisdom that neural network-based models are superior to count-based models.
- While model design is important, hyperparameters are also KEY for achieving reasonable results.  

### Projective

Dependencies parallel to a CFG tree must be projective(must not be any crossing dependency arcs).  
But dependency theory normally does allow non-projective structures to account for displaced constituents.

### Evaluation  

- English parsing to Stanford Dependencies:
	- Unlabeled attachment score (UAS) = head
	- Labeled attachment score (LAS) = head and label

### Indicator Features

Traditional indicator features: sprase, more than 95% of parsing time is consumed by feature computation.  

### Distributed Representations

a d-dimensional dense word vector + part-of-speech tags (POS) and dependency labels are also represented as d-dimensional vectors

Hidden layer h: $h = ReLU(Wx + b_1)$  
Output layer y: $y = softmax(Uh + b_2)$  

