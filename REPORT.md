# Pythorch，一个神经网络编译器/运行时

## 实验目标

搭建一套编译工具以支持将简单的Pytorch CNN模型迁移到x86/ARM Cortex-M架构的嵌入式设备运行，探索各种加速手段

## 构建开发/测试流程

由于条件所限，项目需要在Darwin/Linux系统上进行。因此，我们首先将项目资料中基于Visual Studio的代码转换成CMake工程的。这一过程中，我们创建了`CMakeLists.txt`，并进行了模块的重新划分。

- `MNIST_PYTHORCH_C/cc`保存神经网络的模块/通用函数
- `MNIST_PYTHORCH_C/python` 一个`编译器`，生成C函数
- `tests` 一些测试工程
- `ci` 自动测试脚本

我们首先根据示例代码，构建测试流程。测试脚本为位于`./ci`目录下的`bootstrap.sh`。测试大体上分为两个阶段：

1. 借助PyTorch训练模型，验证模型的准确率。导出C代码
2. 使用导出的代码编译

- 也可以使用`build_cmake.sh`单独测试C推理代码的功能

我们需要在电脑上安装torch，matplotlib，ipython等Python软件包来训练/导出模型。另外，项目生成的C代码默认使用clang进行编译，但也可以用GCC编译。

## 问题约束

我们的理想是设计一套将任意Pytorch模型转换为C代码的机制。然而Pytorch采取了动态计算图（Eager Mode)。在动态计算图中，参与计算的元素不仅有Pytorch Module，还包括了Pytorch Function、Python对象甚至是Numpy数组。Pytorch动态图的这种特性对语法解析工作提出了巨大挑战。考虑到项目时间有限，不可能覆盖针对种种复杂情况进行测试。我们决定将问题约束如下：

1. 转换的对象是`torch.nn.Sequential`模型。这种模型是一系列模块的顺序组合
2. 转换仅支持这些模块`nn.Conv2D`,`nn.Linear`,`nn.MaxPool2d`,`nn.AvgPool2d`,`nn.ReLU`,`nn.Flattern`,`nn.Sigmoid`,`nn.Tanh`。其中`nn.AvgPool2d`,`nn.Sigmoid`,`nn.Tanh`优先级次之
3. 只考虑float32类型数据的计算

以课程示例例为例。示例中的模型可以用nn.Sequenctial包装如下

```python
nn.Sequential(
    nn.Conv2d( 1, 32, 5, 1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 32, 5, 1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(1),
    nn.Linear(512, 1024),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(1024, 10),
)
```

我们也沿用了示例代码中将测试数据保存在`test_data.c/h`文件中的做法。这主要是为了测试的便利性。在模型的实际的运用中，图像数据应该从摄像头模块获取。

## 编译Pytorch模型为C代码

示例中给出了一种将Pytorch模型转换为C代码的方法：

- 模型权重被导出到`export_code/param.c`中，并在`export_code/param.h`声明
- 模型的部分测试数据被导出到`export_code/test_data.c`中，并被编译进了程序

我们按照这个思路，对示例代码进行了扩展。构造的编译器可以一次分析`nn.Sequential`模型的各个子模型，通过正则表达式将它们与对应的模版匹配。并将子模块路由到相应的处理模块进行处理。这里以最复杂的`Conv2d`模块的处理为例。该模块的对应处理函数如下

```python
    @classmethod
    def get_conv2d(cls, context, component: nn.Module):
        """处理Conv2d模块

        Args:
            context (_type_): 编译上下文
            component (nn.Module): Conv2d模块

        Returns:
            CompilerToken: CompilerToken
        """

        layer_id: str = context["id"]
        op: str = "conv2d" + "_" + context["dtype"]  # PYTHORCH_CONV2D_OP
        in_num_c, in_hgt, in_wid = context["in_shape"][
            1:4
        ]  # PYTHORCH_CONV2D_ID_IN_HGT, PYTHORCH_CONV2D_ID_IN_WID
        # PYTHORCH_CONV2D_ID_WEIGHT
        # PYTHORCH_CONV2D_ID_BIAS
        # PYTHORCH_CONV2D_ID_SHAPE
        dout = context["dout"]
        din = context["din"]

        state_dict = component.state_dict()

        # 计算需要给im2col保留的缓冲区大小
        pad = 0
        stride = 1
        k_hgt, k_wid = state_dict["weight"].shape[2:4]  # 卷积核
        height_col = (in_hgt + 2 * pad - k_hgt) / stride + 1
        width_col = (in_wid + 2 * pad - k_wid) / stride + 1
        channels_col = in_num_c * k_hgt * k_wid
        dbuf = context["app"] + cls.buf_postfix
        dbuf_sz = height_col * width_col * channels_col

        free_variables = [context["din"]]
        context["din"] = context["dout"]
        context["dout"] = ""

        res = f"{op}({dout}, {din}, {in_hgt}, {in_wid}, {op}_{layer_id}_weight, {op}_{layer_id}_bias, {op}_{layer_id}_shape, {dbuf});"
        return CompilerToken(
            c_fn=res,
            c_params={
                f"{op}_{layer_id}_weight": state_dict["weight"],
                f"{op}_{layer_id}_bias": state_dict["bias"],
                f"{op}_{layer_id}_shape": state_dict["weight"].shape,
            },
            out_shape=cls.get_out_shape(component, context["in_shape"]),
            free_vars=free_variables,
            buf_claim=dbuf_sz,
        )
```

最终该函数将会生成一行C语句、记录Conv2d的权重、计算Conv2d的输出尺寸，输出Conv2d释放的变量名称和需求的临时变量大小。

值得注意的是，我们不需要处理Flattern，因为Flattern是对张量形状的处理

以此类推，我们处理所有支持的Pytorch模块，最后生成一个这样的函数

```c
#include "pythorch/pythorch.h"
#include "calc_params.h"
__attribute__ ((aligned (32))) float var_0[18432];
__attribute__ ((aligned (32))) float var_1[18432];

pythorch_err_t calc_fn(float* din, float* dout) {

    pythorch_err_t err = PYTHORCH_OK;

    conv2d_f32(var_0, din, 28, 28, conv2d_f32_0_weight, conv2d_f32_0_bias, conv2d_f32_0_shape, calc_g_buf);
    relu_f32(var_1, var_0, 18432);
    maxpool2d_f32(var_0, var_1, 24, 24, 32, 2);
    conv2d_f32(var_1, var_0, 12, 12, conv2d_f32_3_weight, conv2d_f32_3_bias, conv2d_f32_3_shape, calc_g_buf);
    relu_f32(var_0, var_1, 2048);
    maxpool2d_f32(var_1, var_0, 8, 8, 32, 2);
    /** Flatten **/
    linear_f32(var_0, var_1, linear_f32_7_weight, linear_f32_7_bias, linear_f32_7_shape);
    relu_f32(var_1, var_0, 1024);
    /** Dropout **/
    linear_f32(dout, var_1, linear_f32_10_weight, linear_f32_10_bias, linear_f32_10_shape);

    return err;

}
```

其中`pythorch/pythorch`是整个推理库的头文件，引入该头文件并链接推理库就可以使用。该推理库的设计于下一章节叙述。`__attribute__ ((aligned (32))) `是为AVX指令准备的。我们可以看到，自始至终只有两个临时遍历那个var_0和var_1被用来保存的中间结果，这是因为我们处理的是一个序列，并不涉及到复杂的计算图。在复杂的计算图中，可能需要多个变量保存中间结果。


## C神经网络推理库及其优化

我们修改了整理了资料中出现的神经网络算子，将其组织在`MNIST_PYTHORCH_C`目录下：

- `avx_mathfun.h` 使用AVX指令实现的数学函数(sin/cos/exp等)
- `CMakeLists.txt` 定义了pythorch库
- `conv.c/h` conv2d的帮助函数，主要包括im2colsnuff
- `gemm.c/h` 通用矩阵乘法函数
- `moduels.c/h` conv2d/linear/maxpool2d等算子。所有算子都有后缀标明适用的数据类型
- `pythorch.h` 头文件集合
- `utils.c/h` `calc_error`函数，一些自定义类型

具体的加速手段如下

- 对于所有的激活函数，使用AVX指令集进行加速。
- 对于Linear算子的向量点积，使用AVX指令集进行加速。
- 将卷积输入使用im2col算法处理成二维矩阵，将卷积问题转化成GEMM问题
- 针对GEMM操作，运用AVX/分块实现优化
- 对于池化层，采取策略减少数组下标的计算

im2col是将一个`[C,H,W]`矩阵变成一个`[H,W]`矩阵的一个方法，其原理是利用了行列式进行等价转换。im2col需要一个临时空间保存转换后的输入图像。在有MMU的平台上，这段内存可以用malloc动态分配，但是嵌入式设备的动态内存分配可能会导致不稳定。我们选择分配一个固定地址的静态空间作为缓冲。

### 对比测试

对于推理算子的效能分析在一台Hyper-V虚拟机上进行。该虚拟机分配了4GiB内存，4个vCPU(i5-8400)。该型号CPU支持AVX2指令集，即256bit的并行指令。我们在虚拟机上安装了Ubuntu 20.04操作系统，编译器是clang@10.0.0。测试过程中使用`time`命令计时并将输出重定向到文件到文件。

首先，我们在不进行任何优化的情况下以Debug模式编译并执行推理程序。推理500张图片耗时6.705s。

```
real    0m6.710s
user    0m6.705s
sys     0m0.004s
```

我们将CMake工程调整到Release模式，这将启用编译器的优化。推理耗时下降到了1.308s(-5.397)。接下来的实验均在Release模式下进行

```
real    0m1.321s
user    0m1.308s
sys     0m0.012s
```

我们为代码添加AVX支持并通过添加`-mavx,-msse`启用AVX加速。推理耗时下降到了1.023s(-0.285)。

```
real    0m1.023s
user    0m1.023s
sys     0m0.000s
```

我们将卷积操作利用im2col算法转换成矩阵乘法(`-DDOPTIMIZE_IM2COL`)。推理耗时下降到了0.215s(-0.808)。注意，此时的矩阵乘法是较为naive的实现，但也考虑到了访存的优化，调整了循环的次序。

```c
for (int i = 0; i < m_hgt; i++){}
    for (int k = 0; k < n_hgt; k++)
        for (int j = 0; j < n_wid; j++)
            dout[i * n_wid + j] += m[i * m_wid + k] * n[k * n_wid + j];
```

```
real    0m0.220s
user    0m0.215s
sys     0m0.004s
```

在加入GEMM的基础上，我们调整了池化层的算法，尽可能消除了下标的计算。推理耗时下降到了0.206s(-0.009)

```
real    0m0.209s
user    0m0.206s
sys     0m0.004s
```

我们对卷积过程中出现的矩阵乘法使用AVX指令进行优化(`-DDOPTIMIZE_GEMM=1`)。推理耗时下降到了0.142s(-0.009)
开AVX + IM2COL + 消除下标 + GEMM 优化(向量化)

```
real    0m0.142s
user    0m0.142s
sys     0m0.000s
```

进一步考虑到缓存的局部性，我们使用所有16个YMM寄存器，将矩阵乘法拆成8x8的小块进行计算(`-DDOPTIMIZE_GEMM=2`)。推理耗时下降到了0.127s(-0.015)

```
real    0m0.127s
user    0m0.127s
sys     0m0.000s
```

至此，面向x86平台的SIMD优化过程告一段落。我们将程序的执行耗时从6.705s缩短到0.127s。其中，比较有效的方法是开启编译器优化、使用IM2COL计算卷积、使用AVX指令这三种。

## 在IoT-Lab测试

![ST B-L475E-IOT01A](img/20220524104119.png)

![CubeMX](img/20220524104544.png)

![Helloworld](img/20220524113623.png)


## Reference

[The Very Large Scale IoT Testbed](https://www.iot-lab.info/)
[Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
[通用矩阵乘法及其优化](https://lzzmm.github.io/2021/09/10/GEMM/)