import lyutorch.Variables.tensor_math as tm
from .tensor import Tensor

setattr(Tensor, "__add__", tm.__add__)
setattr(Tensor, "__matmul__", tm.__matmul__)
