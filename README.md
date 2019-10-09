# DCCA: Deep Canonical Correlation Analysis

This is an implementation of Deep Canonical Correlation Analysis (DCCA or Deep CCA) in Python with pytorch, which supports for multi-GPU training.

DCCA is a non-linear version of CCA which uses neural networks as the mapping functions instead of linear transformers. DCCA is originally proposed in the following paper:

Galen Andrew, Raman Arora, Jeff Bilmes, Karen Livescu, "[Deep Canonical Correlation Analysis.](http://www.jmlr.org/proceedings/papers/v28/andrew13.pdf)", ICML, 2013.

It uses the latest pytorch1.0-preview. Because the loss function of the network needs to calculate the gradient of eigenvalue decomposition for symmetric matrix. The base modeling network can easily get substituted with a more efficient and powerful network like CNN.

Most of the configuration and parameters are set based on the following paper:

Weiran Wang, Raman Arora, Karen Livescu, and Jeff Bilmes. "[On Deep Multi-View Representation Learning.](http://proceedings.mlr.press/v37/wangb15.pdf)", ICML, 2015.

### Prerequest

* python 3.6+
* pytorch 1.0+ (Supports for calculating the gradient of eigenvalue decomposition for symmetric matrix (symeig))

You can also install the environment as the following:
```bash
conda create -f requirement.yml
conda activate deepcca
```

### Dataset
The model is evaluated on a noisy version of MNIST dataset. I use the dataset built by @VahidooX which is exactly like the way it is introduced in the paper. The train/validation/test split is the original split of MNIST.

The dataset was large and could not get uploaded on GitHub. So it is uploaded on another server. You can download them from [noisymnist_view1.gz](https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view1.gz) and [noisymnist_view2.gz](https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view2.gz), or use the download_data.sh. (Thanks to @VahidooX)

### Differences with the original paper
The following are the differences between my implementation and the original paper (they are small, just same as @VahidooX's implementation):

 * I used RMSProp (an adaptive version of gradient descent) instead of GD with momentum. It was so much faster in converging.
 * Instead of a non-saturating version of sigmoid, I just used the standard sigmoid as the activation functions. Standard sigmoid is used in the MATLAB implementation too. It should not affect the performance significantly. However, if it is needed, it can get substituted by another non-saturating activation function like ReLU.
 * Pre-training is not done in this implementation. However, it is not clear how much it can be useful.

### Other Implementations
The following is the implementation of DCCA in keras with theano backend from which I got great help for the implementation. The codes are written by @VahidooX:

* [Keras implementation](https://github.com/VahidooX/DeepCCA) from @VahidooX's github page(https://github.com/VahidooX)

The following are the other implementations of DCCA in MATLAB and C++. These codes are written by the authors of the original paper:

* [C++ implementation](https://homes.cs.washington.edu/~galen/files/dcca.tgz) from Galen Andrew's website (https://homes.cs.washington.edu/~galen/)

* [MATLAB implementation](http://ttic.uchicago.edu/~wwang5/papers/dccae.tgz) from Weiran Wang's website (http://ttic.uchicago.edu/~wwang5/dccae.html)
