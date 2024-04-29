import numpy as np

import lyutorch as lyu
import lyutorch.Variables.utils as utils
import lyutorch.tmath as tm
from lyutorch import Tensor


def softmax(x: Tensor, target: Tensor):
    """
    This function is used to compute the softmax of the input tensor.

    Parameters
    ----------
    x : (...,Dims) Tensor
        The vectors to be softmaxed. Normally it should be (Batch_size, Dims)
    target: (Dims, Dims) Tensor
        The target of the softmax.
    """
    x = x @ target.t()
    x = tm.exp(x)
    z = tm.sum(x, axes=-1, keepdims=True)
    x = x / z
    return x


def relu(x: Tensor):
    def relu_grad(
        self: Tensor,
        prev_self: Tensor,
        pass_in_grad,
        pass_in: bool = True,
    ):
        grad = np.eye(self.size).reshape(self.shape * 2) * (prev_self > 0)
        utils.make_grad(self, pass_in_grad, pass_in, grad, 0, prev_self)

    ret = Tensor(np.maximum(x, 0), requires_grad=x.requires_grad)
    utils.attach_backward_fn(ret, x.requires_grad, relu_grad, x)
    return ret


def cross_entropy_loss(y_pred: Tensor, y_true: Tensor):
    ret = tm.log(y_pred) * y_true * Tensor(-1)
    ret = lyu.squeeze(tm.sum(ret) * Tensor(1 / y_pred.shape[0]))
    return ret
