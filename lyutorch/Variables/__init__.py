import lyutorch.Variables.tensor_math as tm
from .tensor import Tensor

__all__ = ["Tensor"]

setattr(Tensor, "__add__", tm.__add__)
setattr(Tensor, "__matmul__", tm.__matmul__)
setattr(Tensor, "__mul__", tm.__mul__)
setattr(Tensor, "t", tm.t)
