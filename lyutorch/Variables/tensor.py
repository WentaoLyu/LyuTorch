import numpy as np

from .tape import tape


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
        self.grad = 0 if self.requires_grad else None
        self.backward_fn = None

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
            self.grad = (
                np.eye(self.size).reshape(self.shape * 2)
                if self.ndim > 0
                else np.asarray(1)
            )
        else:
            raise ValueError(
                "Cannot call backward on a tensor that does not require grad."
            )
        if tape.graph_exists is False:
            raise ValueError(
                "Cannot call backward on a tensor that does not have a graph."
            )
        topo_order = tape.topo_sort()
        topo_order.reverse()
        for tensor in topo_order:
            if hasattr(tensor, "backward_fn") and (tensor.backward_fn is not None):
                tensor.backward_fn(tensor.grad, True)
                tensor.backward_fn = None
        tape.clear()

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other
