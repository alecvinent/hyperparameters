
##Automatically Tuning Hyperparameters Is A Dream

##Why
Hyperparameter is controling how to learn the optimization algorithm. So it could directly effect the convergence performence as well as model precision. Given well tuned hyperparameters, even a simple model could be robust enough. Check the publication of "Bayesian Optimization of Text Representations". According to experiences, the optimization alogrithm is very sensitive to learning rate and regularization parameters.

##Idea
Firstly, I deal with this problem from two individual spaces, one is the parameter, the other is the hyper-parameter. Learning could be consided as picking one point from the HPS(hyper parameters space) and then getting training result from the paramter space. How to map the two different spaces and pick an optimized point from HPS via the performance of parameter space? Researchers have found that
reverse-mode differentiation proposed by Bengio(2000) in his paper "Gradient-based optimization of hyperpa-parameters" could resolve this issue. But there exists a big problem with RMD, it will consume thousands of times of memory to store the reverse path. To solve this problem the paper "Gradient-based Hyperparameter Optimization through Reversible Learning", which relies on momentum could reduce hunderds of times of memory compared with the origin RMD. Jie Fu's paper "DrMAD: Distilling Reverse-Mode Automatic Differentiation for Optimizing Hyperparameters of Deep Neural Networks" discards all training trajectories with zero memory consumption.


##Current Implemention
Mixing [autograd](https://github.com/HIPS/autograd) and mxnet to implement DrMAD, having been tested on a small data set mnist.

##Note
This project is an experiment to find a solution to automatically tune hyperparameters.

##How to run it
1. Clone [mxnet](https://github.com/deepinsight/mxnet.git) under deepinsight, then check to branch `hypergrad` and install.
2. Run ```python setup.py``` in ```hyperparameters/mxnet_autograd```.
3. Run ```python train_mnist.py``` in ```hyperparameters/mxnet_autograd/image-classification```.

