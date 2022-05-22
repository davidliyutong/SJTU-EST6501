# 神经网络运算优化

## Target

目标：搭建一套编译工具以支持将简单的Pytorch CNN模型迁移到ARM Cortex-M架构的嵌入式设备运行
进阶目标：在完成主要目标的基础上，探索各种加速手段

预计支持的算子

- Conv2D
- Linear
- MaxPool2d
- ReLU
- Flattern
- ArgMax

未来

- AvgPool2d
- Sigmoid
- Tanh

## RoadMap

- [ ] 完成VS项目到CMake项目的迁移
- [ ] 完成x86_64架构下的测试
- [ ] 完成QEMU虚拟机下的ARM测试
- [ ] 搭建

## 构建测试流程

我们首先根据示例代码，构建持续集成测试

## 问题约束

我们的理想是设计一套将任意Pytorch模型转换为C代码的机制。然而Pytorch采取了动态计算图（Eager Mode)。在动态计算图中，参与计算的元素不仅有Pytorch Module，还包括了Pytorch Function、Python对象甚至是Numpy数组。Pytorch动态图的这种特性对语法解析工作提出了巨大挑战。考虑到项目时间有限，不可能覆盖针对种种复杂情况进行测试。我们决定将问题约束如下：

1. 转换的对象是`torch.nn.Sequential`模型。这种模型是一系列模块的组合
2. 转换仅支持这些模块`nn.Conv2D`,`nn.Linear`,`nn.MaxPool2d`,`nn.AvgPool2d`,`nn.ReLU`,`nn.Flattern`,`nn.Sigmoid`,`nn.Tanh`。其中`nn.AvgPool2d`,`nn.Sigmoid`,`nn.Tanh`优先级次之
3. 只考虑float/int16类型数据的计算
4. 不考虑argmax函数

以课程示例例为例。示例中的模型可以表示如下

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

## 编译Pytorch模型为C代码

示例中给出了一种将Pytorch模型转换为C代码的方法：

- 模型权重被导出到`export_code/param.c`中，并在`export_code/param.h`声明
- 模型的部分测试数据被导出到`export_code/test_data.c`中，并被编译进了程序
