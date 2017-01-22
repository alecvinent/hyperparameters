
##Automatic tuning hyperparameters is a dream.

##Why ?
Hyperparameter is controling how the learning algorithm to learn. so it's directly effect the performence.
With a well tuned hyperparameters, a simple module will be very robust, check paper
"Bayesian Optimization of Text Representations". And on the experience, the learning alogrithm is so sensitive
to learning rate and regularization.

##Idea,
First, I treat this problem as two different sapce, one is the parameter, another is the hyper parameter. Learning
can be consided as pick one point in the HPS(hyper parameters space), to get a training result in the paramter space.
How to mapping the two different space, and pick a optimize point in HPS val parameter space performence? research found
reverse-mode differentiation was pro-posed by Bengio(2000)"Gradient-based optimization of hyperpa-parameters" can
solve this issue.But there is a big problem with RMD, it will comsume thousands of times memory to storage the reverse path.
To solve this problem paper "Gradient-based Hyperparameter Optimization through Reversible Learning", rely on moment can reduce hunderd times
memory compare to origin RMD. Jie Fu's paper "DrMAD: Distilling Reverse-Mode Automatic Differentiation for Optimizing
Hyperparameters of Deep Neural Networks" throw all training trajectory, with zero memory comsume.


##current implement
mix augograd and mxnet to implement DrMAD, tested on small data set mnist.

##note
this project is a experiment project to find a solution to automatic tuning hyperparameters.

