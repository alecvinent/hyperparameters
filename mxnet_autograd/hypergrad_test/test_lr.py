from .nn_utils import make_nn_funs, VectorParser
import numpy as np
from core import grad
#from .optimizers import sgd_numpy_safe as sgd
#import logging

def make_nn(arg_shapes):
    #[(128L, 1L, 28L, 28L), (128L, 784L), (128L,), (64L, 128L), (64L,), (10L, 64L), (10L,), (128L,)]
    layers_mxnet = arg_shapes[0]

    layer_autograd = []

    #first layer is the minidata batch shape which formed as data recorde,
    #second layer is the minidata batch shape which has been falted.k
    #last layer is for input data block shape, it isn't need for auto_grad
    #strip the first and the last layer, which isn't need.
    for i, layer in enumerate(layers_mxnet[1:-1]):
        # data input shape
        if i == 0:
            layer_autograd.append(layer[1])
        # if the layer is active layer, we can get number of neurons in the current layer
        if len(layer) == 1:
            layer_autograd.append(layer[0])

    parser, predictions, loss, frac_err = make_nn_funs(layer_autograd)
    return parser, predictions, loss, frac_err, len(layer_autograd)

def mxnet_weight2autograd_wieght(mw, aw, layer_num):
    #arg_params['fc1_weight'].asnumpy().T.shape

    vect = np.zeros((0,))

    for i in range(layer_num - 1):
        cur_W_shape = aw.idxs_and_shapes[('weights', i)][1]
        cur_B_shape = aw.idxs_and_shapes[('biases', i)][1]
        cur_B_shape = (cur_B_shape[1], )


        for key in mw:
            value = mw[key]
            if len(value.shape) == 2:
                value_t = value.T
            else:
                value_t = value
            if value_t.shape == cur_W_shape:
                vect = np.concatenate((vect, value_t.asnumpy().flatten()), axis=0)

        for key in mw:
            value = mw[key]
            if value.shape == cur_B_shape:
                vect = np.concatenate((vect, value.asnumpy()), axis=0)

    return vect


def grad_lr(mxnet_weight, autograd_weight, test_images, test_labels, layer_num, loss_fun, w_init, alpha):
    agw = mxnet_weight2autograd_wieght(mxnet_weight, autograd_weight, layer_num)
    loss_grad = grad(loss_fun)

    one_hot = lambda x, K : np.array(x[:,None] == np.arange(K)[None, :], dtype=int)
    partial_flatten = lambda x : np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))

    test_images = partial_flatten(test_images)  / 255.0
    test_labels = one_hot(test_labels, 10)
    train_mean = np.mean(test_images, axis=0)
    test_images = test_images - train_mean

    g = loss_grad(agw, test_images, test_labels)
    d_v = np.zeros(g.shape)

    d_v += g * alpha
    d_alpha = np.dot(g, d_v)
    return d_alpha

