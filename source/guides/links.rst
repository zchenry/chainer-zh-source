链接
~~~~

为了描述神经网络，我们需要结合函数于 *参数* 并优化这些参数。
你可以通过使用 class :class:`Link` 来实现这个目的.
:class:`Link` 是一个保持参数的对象。

最基本的链接是表现为普通函数的链接，但是同时由参数替代一些表达式。
我们会介绍一些更高级的链接，但是现在你可以把链接当成拥有参数的简单函数。

最常用的链接之一是 :class:`~functions.Linear` 链接 (也就是 *全连接层* 或 *affine 转换*).
它表达了函数 :math:`f(x) = Wx + b`, 行列 :math:`W` 和向量 :math:`b` 是参数。
这个链接和它的单纯的对于函数 :func:`~functions.linear` 有关，该函数只接受参数 :math:`x, W, b` 。
一个从三次空间到二次空间的线性链接可以如下定义：

.. doctest::

   >>> f = L.Linear(3, 2)

.. note::

   大部分函数和链接只接受小样本输入，其第一次元通常是 *小样本长度*。
   在如上的线性链接中，输入必须有形状 :math:`(N, 3)` ，:math:`N` 是小样本长度。

链接的参数被作为属性保存。
每个参数都是 :class:`~chainer.Variable` 的一个实现。
线性链接的场合，两个参数 ``W`` 和 ``b`` 被保存。
默认下，行列 ``W`` 被随机初始化，向量 ``b`` 由初始化为零向量。
这是合适的初始化方法。

.. doctest::

   >>> f.W.data
   array([[ 1.0184761 ,  0.23103087,  0.5650746 ],
          [ 1.2937803 ,  1.0782351 , -0.56423163]], dtype=float32)
   >>> f.b.data
   array([0., 0.], dtype=float32)

一个线性链接的实现可被用做一个普通的函数:

.. doctest::

   >>> x = Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
   >>> y = f(x)
   >>> y.data
   array([[3.1757617, 1.7575557],
          [8.619507 , 7.1809077]], dtype=float32)

.. note::

  有时计算输入空间的次元会很麻烦。
  线性链接和一些（反）卷积层链接可以在初始化时省略输入的次元并从第一个小样本中推定。

  比如，下面的一行代码创造了一个输出次元是 2 的线性链接::

  >>> f = L.Linear(2)

  如果我们输入一个形状为 :math:`(2, M)` 的小样本，输入次元会被推定为 ``M`` ，
  这意味这 ``l.W`` 会是一个 2 x M 行列。
  注意参数会在接收第一个小样本的时候初始化。
  因此，在没有数据输入的时候 ``l`` 不会有 ``W`` 属性。

参数的微分可由 :meth:`~Variable.backward` 计算。
注意微分会被 **累积** 而不是覆盖。
所以你首先要清除掉微分。
这可以由执行 :meth:`~Link.cleargrads` 来实现。

.. doctest::

   >>> f.cleargrads()

.. note::
   :meth:`~Link.cleargrads` 由 v1.15 导入以取代 :meth:`~Link.zerograds` 以提高效率。
   :meth:`~Link.zerograds` 只为了向后兼容而留存。

现在我们可以简单地计算参数的微分并通过 ``grad`` 属性来表示微分。

.. doctest::

   >>> y.grad = np.ones((2, 2), dtype=np.float32)
   >>> y.backward()
   >>> f.W.grad
   array([[5., 7., 9.],
          [5., 7., 9.]], dtype=float32)
   >>> f.b.grad
   array([2., 2.], dtype=float32)
