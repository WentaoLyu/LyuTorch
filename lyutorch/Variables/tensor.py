import numpy as np


class Tensor(np.ndarray):
    def __new__(
        cls, input_data, requires_grad=False, dtype=None, order=None, *args, **kwargs
    ):
        obj = np.asarray(input_data, dtype=dtype, order=order, *args, **kwargs).view(
            cls
        )
        obj.requires_grad = requires_grad
        obj.grad = 0 if requires_grad else None
        obj.backward_fn = None
        obj._PASS_IN = True
        if requires_grad:
            obj._prev = []
        else:
            obj._prev = None
        return obj

    def __array_finalize__(self, obj, *args, **kwargs):
        if obj is None:
            return
        self.requires_grad = getattr(obj, "requires_grad", False)

    def __array_wrap__(self, out_arr, context=None):
        out_arr = out_arr.view(Tensor)
        out_arr.requires_grad = self.requires_grad
        return out_arr

    def backward(self):
        if self.requires_grad:
            self.grad = np.eye(self.size).reshape(self.shape * 2)
        else:
            raise ValueError(
                "Cannot call backward on a tensor that does not require grad"
            )
        self.backward_fn(self.grad, False)
