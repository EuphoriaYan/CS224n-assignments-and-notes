## Introduction to TensorFlow

### Deep Learning Frameworks

- These frameworks scale machine learning code
- Easy to computes gradients
- Standardizes machine learning applications for sharing
- Zoo（融合） of Deep Learning frameworks available with different advantages, paradigms, levels of abstraction, programming languages, etc
- Interface with GPUs for parallel processing

### Variable sharing

When you want to build a large model, you often need to share large sets of variables;  
Instantiate a graph multiple times/ train over a clusters of GPUs.  

Naive way: create variables' dictionary. Not good for encapsulation.  
The code that your graph's intensive flow should always have all of the relevant information about the nodes and operations that you are using.(Using dictionary will loss information about the document of operations & the shape of your variables)  

Solution: Variable scope  

	tf.variable_scope()
	tf.get_variable()

It provides a simple name spacing scheme to avoid clashes.  The function ```get_variable``` will create a variable if it doesn't exist, or access a variable which has this name.  

	with tf.variable_scope("f"):
		v = tf.get_variable("v", shape = [1])	# v.name == "f/v:0"
	with tf.variable_scope("f", reuse=True):
		v = tf.get_variable("v")	# find shared variable
	with tf.variable_scope("f", reuse=False):
		v = tf.get_variable("v")	# Error, "f/v:0" already exist


