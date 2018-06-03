
   在此教程的样例代码中，为了简洁我们假设以下库已被导入。

.. testcode::

     import numpy as np
     import chainer
     from chainer.backends import cuda
     from chainer import Function, gradient_check, report, training, utils, Variable
     from chainer import datasets, iterators, optimizers, serializers
     from chainer import Link, Chain, ChainList
     import chainer.functions as F
     import chainer.links as L
     from chainer.training import extensions

