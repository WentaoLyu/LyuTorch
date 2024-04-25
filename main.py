"""
本次作业使用了 LyuTorch，其实现了一个 Tensor 类，这是一个 numpy.ndarray 的子类，在这个类中，作者手动实现了反向求导的功能.
Tensor 类支持对少数操作的反向求导，足以构造一个只包含全链接层神经网络，方法十分简单
>>> import lyutorch as lyu
>>> a = lyu.tensor([[1, 2, 3]], requires_grad=True)
>>> c = a @ a.t()
>>> c = lyu.squeeze(c)
>>> c.backward()
>>> print(a.grad)
[[1. 2. 3.]]

上方的例子展示了如何使用 LyuTorch 进行反向求导，该例子计算了一个向量和自身的点乘.
这里的 c = a @ a.t() 是矩阵乘法，squeeze 是压缩维度，去掉长度为 1 的维度，backward 是反向求导.
"""
