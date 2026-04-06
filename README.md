### a step by step implementation for a series of softmax 

to understand flash attention, you must fully understand and learn to derive softmax by yourself. this is a series of experiments to show different and optimized
variations of softmax. we will do all of it in pytorch.

this is how pytorch defines softmax as -

$\mathrm{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$

1. `01-softmax.py` - this version is the simplest naive representation to perform softmax operation. it should help you implement softmax step by step.