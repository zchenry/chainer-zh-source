.. _install-guide:

安装
====

推荐环境
--------

我们推荐如下的 Linux 发行版.

* `Ubuntu <https://www.ubuntu.com/>`_ 14.04 / 16.04 LTS (64-bit)
* `CentOS <https://www.centos.org/>`_ 7 (64-bit)

.. note::

   我们在上述的推荐环境下对 Chainer 做自动测试.
   我们不保证 Chainer 在其他环境下的运行，包括 Windows 和 macOS (特别是有 CUDA 支持的), 尽管 Chainer 也许会正确地运行。


需求
----

你需要如下组件来使用 Chainer。

* `Python <https://python.org/>`_
    * 支持版本: 2.7.6+, 3.4.3+, 3.5.1+ 和 3.6.0+.
* `NumPy <http://www.numpy.org/>`_
    * 支持版本: 1.9, 1.10, 1.11, 1.12 and 1.13.
    * NumPy 将会在 Chainer 安装时自动安装。

安装 Chainer 前，我们推荐你升级 ``setuptools`` 和 ``pip``::

  $ pip install -U setuptools pip

关于硬件加速的支持
~~~~~~~~~~~~~~~~~~

你能安装如下可选组件来加速 Chainer 的运行。

* NVIDIA CUDA / cuDNN
    * `CuPy <https://cupy.chainer.org/>`_ 4.0+
    * 详见 `CuPy Installation Guide <https://docs-cupy.chainer.org/en/latest/install.html>`__ 。

* Intel CPU (实验性的)
    * `iDeep <https://github.com/intel/ideep>`_ 1.0.3+
    * 详见 :doc:`tips` 。

可选功能
~~~~~~~~

如下的安装包是可选的。
Chainer 能在缺少这些安装包的情况下安装, 但是相关功能会无法使用。

* 图像数据支持
    * `pillow <https://pillow.readthedocs.io/>`__ 2.3+
    * 运行 ``pip install pillow`` 安装。
* HDF5 线性支持
    * `h5py <http://www.h5py.org/>`__ 2.5+
    * 运行 ``pip install h5py`` 安装。


安装 Chainer
------------

使用 pip
~~~~~~~~

我们推荐使用 pip 安装 Chainer::

  $ pip install chainer

.. note::

   任何可选组件可在安装 Chainer 后添加。
   Chainer 自动检测可用组件并且启用相关功能。

使用 Tarball
~~~~~~~~~~~~

源代码的 tarball 可由 ``pip download chainer`` 或从 `发行页面 <https://github.com/chainer/chainer/releases>`_ 获取。
你可以使用 tarball 安装 Chainer::

  $ pip install chainer-x.x.x.tar.gz

你也可以克隆Git库安装开发版本的 Chainer::

  $ git clone https://github.com/chainer/chainer.git
  $ cd chainer
  $ pip install .

启用 CUDA/cuDNN 支持
~~~~~~~~~~~~~~~~~~~~

为了启用 CUDA 支持, 你需要手动安装 `CuPy <https://cupy.chainer.org/>`_ 。
如果你想使用 cuDNN，你需要安装 cuDNN 支持的 CuPy 。
详见 `CuPy 的安装指南 <https://docs-cupy.chainer.org/en/latest/install.html>`__ 。
CuPy 正确安装后, Chainer 将自动启用 CUDA 支持.

你可以引用如下变量来确认 CUDA/cuDNN 支持是否可以启用。

``chainer.backends.cuda.available``
   ``True`` 如果 Chainer 成功启用 :mod:`cupy`.
``chainer.backends.cuda.cudnn_enabled``
   ``True`` 如果 cuDNN 支持可以启用。


卸载 Chainer
------------

使用 pip 卸载 Chainer::

  $ pip uninstall chainer

.. note::

   当你升级 Chainer, ``pip`` 有时会在安装新版本时留存旧版本在 ``site-packages`` 中。
   这种情况下, ``pip uninstall`` 只会删除最新版本。
   为了确保 Chainer 被完全删除, 重复运行上面的命令直到 ``pip`` 报错。


升级 Chainer
------------

使用 ``pip`` 的 ``-U`` 功能::

  $ pip install -U chainer


重新安装 Chainer
----------------

如果你想重新安装 Chainer, 请先卸载 Chainer 然后再安装。
我们推荐使用 ``--no-cache-dir`` 因为 ``pip`` 有时会使用缓存::

  $ pip uninstall chainer
  $ pip install chainer --no-cache-dir


在 Docker 中运行 Chainer
------------------------

我们提供官方的 Docker image。
使用 `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`_ 命令来在GPU环境下运行 Chainer image。
你可以进入该环境使用 bash 运行 Python 解释器::

  $ nvidia-docker run -it chainer/chainer /bin/bash

或者直接运行解释器::

  $ nvidia-docker run -it chainer/chainer /usr/bin/python


常见问题
--------

警告信息 "cuDNN is not enabled"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

你没有成功使用 cuDNN 构建 CuPy。
如果你不需要 cuDNN，忽略这条信息。
否则的话，重新尝试用 cuDNN 安装 CuPy。
``pip install -vvvv`` 命令会帮助到你。
没有必要来重新安装 Chainer。
详见 `CuPy 安装指南 <https://docs-cupy.chainer.org/en/latest/install.html>`__ 。

CuPy 经常发生 ``cupy.cuda.compiler.CompileException``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

详见 `CuPy 安装指南 <https://docs-cupy.chainer.org/en/latest/install.html>`__ 的常见问题。

h5py 安装失败
~~~~~~~~~~~~~

如果安装失败并报错 ``hdf5.h is not found``, 你需要首先安装 ``libhdf5`` 。
安装方法取决于你的环境::

  # Ubuntu 14.04/16.04
  $ apt-get install libhdf5-dev

  # CentOS 7
  $ yum -y install epel-release
  $ yum install hdf5-devel

注意只在需要 HDF5 支持时 ``h5py`` 才是必须的。
