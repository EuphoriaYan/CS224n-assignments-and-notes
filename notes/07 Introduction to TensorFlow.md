## Introduction to TensorFlow

### Deep Learning Frameworks

- These frameworks scale machine learning code
- Easy to computes gradients
- Standardizes machine learning applications for sharing
- Zoo（融合） of Deep Learning frameworks available with different advantages, paradigms, levels of abstraction, programming languages, etc
- Interface with GPUs for parallel processing

### Variable sharing

When you want to build a large model, you often need to share large sets of variables;  
Instantiate a graph multiple times/ train over a clusters of GPUs  

生成一张图的多个实例，或者在多机/多个GPU上训练同一个模型；  
如何在不同位置共享同一个变量？  

Naive way: create variables' dictionary. Not good for encapsulation.  
The code that your graph's intensive flow should always have all of the relevant information about the nodes and operations that you are using.(Using dictionary will loss information about )  




