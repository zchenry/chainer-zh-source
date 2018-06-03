变量(Variable)和导数
~~~~~~~~~~~~~~~~~~~~

.. include:: ../imports.rst

就像之前陈述的，Chainer 使用“运行即定义”模式，所以前反馈计算自身即 *定义* 了网络。
为了开始前反馈计算，我们需要设定输入行列为一个 :class:`Variable` 对象。
再次我们从只有一个元素的简单 :class:`~numpy.ndarray` 开始:

.. doctest::

   >>> x_data = np.array([5], dtype=np.float32)
   >>> x = Variable(x_data)

一个 Variable 对象可以进行基本的算数运算。
为了计算 :math:`y = x^2 - 2x + 1` ，我们只需写:

.. doctest::

   >>> y = x**2 - 2 * x + 1

结果的 ``y`` 也是一个 Variable 对象，其值可以访问 :attr:`~Variable.data` 属性来提取:

.. doctest::

   >>> y.data
   array([16.], dtype=float32)

``y`` 所保持的不止是结果的值。
它也保持计算的履历 (或可以称为计算图)，可以使我们计算其导数。
这可以由执行 :meth:`~Variable.backward` 函数来实现:

.. doctest::

   >>> y.backward()

该代码运行 *误差反馈*。
然后，导数被计算出并被保存在输入变量 ``x`` 的 :attr:`~Variable.grad` 属性里:

.. doctest::

   >>> x.grad
   array([8.], dtype=float32)

另外，我们也可以对中间变量求导。
注意 Chainer 默认释放中间变量的导数行列来提高内存利用效率。
为了保存导数信息，向反馈函数传递 ``retain_grad`` 参数:

.. doctest::

   >>> z = 2*x
   >>> y = x**2 - z + 1
   >>> y.backward(retain_grad=True)
   >>> z.grad
   array([-1.], dtype=float32)

以上的所有计算都可以一般化到多元素行列输入。
虽然单元素行列自动初始化为 ``[1]`` ，为了进行保持多元素行列的变量的反馈计算，我们必须手动设定 *初始误差* 。
这可以由设定输出变量的 :attr:`~Variable.grad` 属性简单实现:

.. doctest::

   >>> x = Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
   >>> y = x**2 - 2*x + 1
   >>> y.grad = np.ones((2, 3), dtype=np.float32)
   >>> y.backward()
   >>> x.grad
   array([[ 0.,  2.,  4.],
          [ 6.,  8., 10.]], dtype=float32)

.. note::

   许多以 :class:`Variable` 对象为输入的函数被定义在 :mod:`~chainer.functions` 模块中.
   你可以结合函数来实现自动反馈的复杂计算。
