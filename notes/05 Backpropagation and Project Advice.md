## Backpropagation and Project Advice

### Explanation #2 for backprop: “Circuits”  

$$ f(x,y,z)=(x+y)z $$  
$$ q = x + y, \cfrac{\partial q}{\partial x} = 1, \cfrac{\partial q}{\partial y} = 1 $$  
$$ f = qz,  \cfrac{\partial f}{\partial q} = z, \cfrac{\partial f}{\partial z} = q $$  

Recursively walk back through circuit, use chain rule.  

### Explanation #3 for backprop:  The high-level flowgraph

The derivative can be seen like the weight on the directed edge. The variable can be seen like the point on the directed graph.  

Walk back through flowgraph, use chain rule.  

### Explanation #4 for backprop: The delta error signals in real neural nets

Use chain rule on the nenural netword, directly calculate derivatives.  

### Highlight: Facebook Fasttext, Efficient Text Classification  

Bag-of-words: Despite the order of words, put words in a bag.  
Calculate their word vectors' average.  
We can use n-grams to capture sequence information.  

FastText: not neural network, just a look-up table and a simple linear regression.  
Use hierarchical（分层的） softmax instead of a huge softmax layer.  

Summary:  

- FastText is often on par with deep learning classifiers
- FastText takes seconds, instead of days
- Can learn vector representations of words in different languages (with performance
better than word2vec)


