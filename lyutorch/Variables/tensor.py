import numpy as np


class Tensor(np.ndarray):
    def __new__(
        cls, input_data, requires_grad=False, dtype=None, order=None, *args, **kwargs
    ):
        obj = np.asarray(input_data, dtype=dtype, order=order, *args, **kwargs).view(
            cls
        )
        obj.requires_grad = requires_grad
        return obj

    def __init__(self, *args, **kwargs):
        self.graph_exists = True
        self.grad = 0 if self.requires_grad else None
        self.backward_fn = None
        self.prev = [] if self.requires_grad else None

    def __array_finalize__(self, obj, *args, **kwargs):
        if obj is None:
            return
        self.requires_grad = getattr(obj, "requires_grad", False)

    def __array_wrap__(self, out_arr, context=None):
        out_arr = out_arr.view(Tensor)
        out_arr.requires_grad = self.requires_grad
        return out_arr

    def backward(self) -> None:
        if self.requires_grad:
            if self.graph_exists is False:
                raise ValueError(
                    "Cannot call backward twice on a tensor since the graph has been cleared."
                )
            if not self.prev:
                raise ValueError("Cannot call backward on a root tensor.")
            self.grad = np.eye(self.size).reshape(self.shape * 2)
        else:
            raise ValueError(
                "Cannot call backward on a tensor that does not require grad."
            )
        self.backward_fn(self.grad, False)
        self.clear_backward_fn()

    def clear_backward_fn(self) -> None:
        if self.requires_grad and self.prev:
            self.backward_fn = None
            self.graph_exists = False
            for tensor in self.prev:
                tensor.clear_backward_fn()
            self.prev = None
