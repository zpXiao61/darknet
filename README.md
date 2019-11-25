添加了 darknet 的 python 接口，在使用了其它接口进行模型推理的情况下，可以复用 darknet 原生的前后处理接口。

# 改动 #
1. 在头文件 'include/darknet.h' 中 添加了 'bm_get_network_boxes' 和 'bm_load_image_and_resize_to_arr' 两个函数；

2. 添加了源文件 'src/bm_func.c' 并在其中实现了上述两个函数；

3. 在 Makefile 的第60行添加了编译目标 bm_func.o；

4. 添加了脚本 'python/darknet_sail_1682.py' 用于测试。

![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# Darknet #
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).
