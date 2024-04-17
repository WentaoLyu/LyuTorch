import numpy as np

from .tensor import Tensor


def _pass_grad(pass_in_grad, *args):
    """
    Pass the gradient to the previous tensors.

    Parameters
    ----------
    pass_in_grad
        The gradient to be passed.
    args
        The tensor(s) to which the gradient is passed.
    """
    for arg in args:
        if arg.requires_grad and (arg.backward_fn is not None):
            arg.backward_fn(pass_in_grad, True)


def _gradient_add(added_grad, ndim, tensor, bias=0):
    """
    Add the gradient to the tensor, campatible with broadcast.

    Parameters
    ----------
    added_grad
        The gradient to be added.
    ndim
        The dimension of the tensor on which the derivative is performed on.
    bias
        The bias index.
    tensor
        The tensor to which the gradient is added.
    """
    if tensor.requires_grad:
        broadcast_dims, created_dims = _check_broadcast(added_grad, ndim, tensor, bias)
        if created_dims is not None:
            if broadcast_dims:
                added_grad = added_grad.sum(
                    axis=tuple(-(tensor.ndim - dim) for dim in broadcast_dims),
                    keepdims=True,
                )
                added_grad = added_grad.sum(
                    axis=tuple(-(ndim - dim) for dim in created_dims),
                    keepdims=False,
                )
                tensor.grad += added_grad
            else:
                added_grad = added_grad.sum(
                    axis=tuple(-(ndim - dim) for dim in created_dims),
                    keepdims=False,
                )
                tensor.grad += added_grad
        elif broadcast_dims:
            added_grad = added_grad.sum(
                axis=tuple(-(ndim - dim) for dim in broadcast_dims),
                keepdims=True,
            )
            tensor.grad += added_grad
        else:
            tensor.grad += added_grad


def make_grad(
    self: Tensor,
    pass_in: bool,
    pass_in_grad: np.ndarray,
    grad: np.ndarray,
    bias: int,
    *args
):
    for arg in args:
        if pass_in:
            grad = np.tensordot(
                pass_in_grad,
                grad,
                axes=(list(range(-self.ndim, 0)), list(range(self.ndim))),
            )
            _gradient_add(grad, self.ndim, arg, bias)
            _pass_grad(grad, arg)
        else:
            _gradient_add(grad, self.ndim, arg, bias)
            _pass_grad(grad, arg)


def _get_identity_tensor_like(tensor_input):
    result = np.eye(tensor_input.size)
    result = result.reshape(tensor_input.shape * 2)
    return result


def _check_broadcast(grad_tensor, dim_derivative, add_tensor, bias=0):
    """
    Check if the added tensor is broadcasted during computation.

    Parameters
    ----------
    grad_tensor
        The grad tensor to be added
    dim_derivative
        The dimensions of tensor on which the derivative is performed on.
    add_tensor
        The tensor to be added.

    Returns
    -------
    broadcast_dims
        The dimensions that are broadcasted.
    created_dim
        The dimensions that are created during computation.

    Examples
    --------
    >>> a = Tensor([1],requires_grad=True)
    >>> b = Tensor([[1, 2],[1,2]])
    >>> c = a + b
    >>> c.backward()
    Then the gradient of c with respect to itself should be of shape (2,2,2,2).
    >>> _check_broadcast(c.grad, 2, a, 0)
    ([0], [0])
    Which is shown that 1 dimension is broadcasted and 1 dimension is created.
    """
    dim_add = add_tensor.ndim - bias
    dim_derivative = dim_derivative - bias
    shape_grad = (
        grad_tensor.shape[-dim_derivative:]
        if bias == 0
        else grad_tensor.shape[-(dim_derivative + bias) : -bias]
    )
    broadcast_dims = []
    for i in range(dim_add):
        if add_tensor.shape[i] == 1 and shape_grad[-(dim_add - i)] != 1:
            broadcast_dims.append(i)
    if dim_add < dim_derivative:
        created_dim = np.arange(0, dim_derivative - dim_add)
    else:
        created_dim = None
    return broadcast_dims, created_dim
