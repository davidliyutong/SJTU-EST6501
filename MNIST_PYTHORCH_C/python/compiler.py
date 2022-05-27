from email.mime import base
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple, Union
import re
from collections import deque
from copy import deepcopy
import os


class CompilerToken:
    """Compiler 分析Pytorch模型，生成一个Token
    """

    def __init__(
        self,
        c_fn: str,
        c_params: Dict[str, torch.Tensor],
        out_shape: Union[torch.Size, List[int]],
        free_vars: List[str],
        buf_claim: int,
    ) -> None:
        """TOken对象初始化

        Args:
            c_fn (str): 生成的C函数
            c_params (Dict[str, torch.Tensor]): KV形式的模块需要保存的权重和数据
            out_shape (Union[torch.Size, List[int]]): 模块输出尺寸
            free_vars (List[str]): 模块计算完成后释放的变量名称
            buf_claim (int): 模块需要的缓冲区大小（单位：元素）。这部分内存的生命周期限于本模块
        """
        self.c_fn = c_fn
        self.c_params = c_params
        self.out_shape = out_shape
        self.free_vars = free_vars
        self.buf_claim = buf_claim


class CompilerTemplate:
    buf_postfix: str = "_g_buf"

    def __init__(self):
        pass

    @classmethod
    def get_out_shape(
        cls, component: nn.Module, in_shape: Union[torch.Size, List[int]]
    ):
        """计算模块输出尺寸

        模型会建立一个假张量，调用模块的Forward计算一遍，然后查看输出的尺寸

        Args:
            component (nn.Module): 神经网络模块
            in_shape (Union[torch.Size, List[int]]): 输入尺寸

        Returns:
            Union[torch.Size, List[int]]: 输出的尺寸
        """
        with torch.no_grad():
            dummy_input = torch.randn(in_shape)
            dummy_output = component(dummy_input)
            out_shape = dummy_output.shape
        return out_shape

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

    @classmethod
    def get_linear(cls, context, component: nn.Module):
        """处理Linear模块

        Args:
            context (_type_): 编译上下文
            component (nn.Module): Linear模块

        Returns:
            CompilerToken: CompilerToken
        """

        layer_id: str = context["id"]
        op: str = "linear" + "_" + context["dtype"]  # PYTHORCH_LINEAR_OP
        # PYTHORCH_LINEAR_ID_WEIGHT
        # PYTHORCH_LINEAR_ID_BIAS
        # PYTHORCH_LINEAR_ID_SHAPE
        dout = context["dout"]
        din = context["din"]

        state_dict = component.state_dict()

        free_variables = [context["din"]]
        context["din"] = context["dout"]
        context["dout"] = ""

        res = f"{op}({dout}, {din}, {op}_{layer_id}_weight, {op}_{layer_id}_bias, {op}_{layer_id}_shape);"
        return CompilerToken(
            c_fn=res,
            c_params={
                f"{op}_{layer_id}_weight": state_dict["weight"],
                f"{op}_{layer_id}_bias": state_dict["bias"],
                f"{op}_{layer_id}_shape": state_dict["weight"].shape,
            },
            out_shape=cls.get_out_shape(component, context["in_shape"]),
            free_vars=free_variables,
            buf_claim=0,
        )

    @classmethod
    def get_maxpool2d(cls, context, component: nn.Module):
        """处理MaxPool2d模块

        Args:
            context (_type_): 编译上下文
            component (nn.Module): MaxPool2d模块

        Returns:
            CompilerToken: CompilerToken
        """

        op: str = "maxpool2d" + "_" + context["dtype"]  # PYTHORCH_MAXPOOL2D_OP
        num_c, in_hgt, in_wid = context["in_shape"][
            1:4
        ]  # PYTHORCH_MAXPOOL2D_ID_IN_HGT, PYTHORCH_MAXPOOL2D_ID_IN_WID, PYTHORCH_MAXPOOL2D_ID_NUM_C
        k_size: int = component.kernel_size  # PYTHORCH_MAXPOOL2D_ID_K_SIZE 24
        # PYTHORCH_LINEAR_ID_WEIGHT
        # PYTHORCH_LINEAR_ID_BIAS
        # PYTHORCH_LINEAR_ID_SHAPE
        dout = context["dout"]
        din = context["din"]

        free_variables = [context["din"]]
        context["din"] = context["dout"]
        context["dout"] = ""

        res = f"{op}({dout}, {din}, {in_hgt}, {in_wid}, {num_c}, {k_size});"
        return CompilerToken(
            c_fn=res,
            c_params=None,
            out_shape=cls.get_out_shape(component, context["in_shape"]),
            free_vars=free_variables,
            buf_claim=0,
        )

    @classmethod
    def get_avgpool2d(cls, context, component: nn.Module):
        """处理AvgPool2d模块

        Args:
            context (_type_): 编译上下文
            component (nn.Module): AvgPool2d模块

        Returns:
            CompilerToken: CompilerToken
        """

        op: str = "avgpool2d" + "_" + context["dtype"]  # PYTHORCH_AVGPOOL2D_OP
        num_c, in_hgt, in_wid = context["in_shape"][
            1:4
        ]  # PYTHORCH_AVGPOOL2D_ID_IN_HGT, PYTHORCH_AVGPOOL2D_ID_IN_WID, PYTHORCH_AVGPOOL2D_ID_NUM_C
        k_size: int = component.kernel_size  # PYTHORCH_AVGPOOL2D_ID_K_SIZE 24
        # PYTHORCH_LINEAR_ID_WEIGHT
        # PYTHORCH_LINEAR_ID_BIAS
        # PYTHORCH_LINEAR_ID_SHAPE
        dout = context["dout"]
        din = context["din"]

        free_variables = [context["din"]]
        context["din"] = context["dout"]
        context["dout"] = ""

        res = f"{op}({dout}, {din}, {in_hgt}, {in_wid}, {num_c}, {k_size});"
        return CompilerToken(
            c_fn=res,
            c_params=None,
            out_shape=cls.get_out_shape(component, context["in_shape"]),
            free_vars=free_variables,
            buf_claim=0,
        )

    @classmethod
    def get_relu(cls, context, component: nn.Module):
        """处理ReLu模块

        Args:
            context (_type_): 编译上下文
            component (nn.Module): ReLu模块

        Returns:
            CompilerToken: CompilerToken
        """

        op = "relu" + "_" + context["dtype"]  # PYTHORCH_RELU_OP
        size = math.prod(context["in_shape"])  # PYTHORCH_RELU_SIZE
        dout = context["dout"]
        din = context["din"]

        free_variables = [context["din"]]
        context["din"] = context["dout"]
        context["dout"] = ""

        res = f"{op}({dout}, {din}, {size});"
        return CompilerToken(
            c_fn=res,
            c_params=None,
            out_shape=context["in_shape"],
            free_vars=free_variables,
            buf_claim=0,
        )

    @classmethod
    def get_sigmoid(cls, context, component: nn.Module):
        """处理Sigmoid模块

        Args:
            context (_type_): 编译上下文
            component (nn.Module): Sigmoid模块

        Returns:
            CompilerToken: CompilerToken
        """

        op = "sigmoid" + "_" + context["dtype"]  # PYTHORCH_SIGMOID_OP
        size = math.prod(context["in_shape"])  # PYTHORCH_SIGMOID_SIZE
        dout = context["dout"]
        din = context["din"]

        free_variables = [context["din"]]
        context["din"] = context["dout"]
        context["dout"] = ""

        res = f"{op}({dout}, {din}, {size});"
        return CompilerToken(
            c_fn=res,
            c_params=None,
            out_shape=context["in_shape"],
            free_vars=free_variables,
            buf_claim=0,
        )

    @classmethod
    def get_tanh(cls, context, component: nn.Module):
        """处理Tanh模块

        Args:
            context (_type_): 编译上下文
            component (nn.Module): Tanh模块

        Returns:
            CompilerToken: CompilerToken
        """

        op = "tanh" + "_" + context["dtype"]  # PYTHORCH_TANH_OP
        size = math.prod(context["in_shape"])  # PYTHORCH_TANH_SIZE
        dout = context["dout"]
        din = context["din"]

        free_variables = [context["din"]]
        context["din"] = context["dout"]
        context["dout"] = ""

        res = f"{op}({dout}, {din}, {size});"
        return CompilerToken(
            c_fn=res,
            c_params=None,
            out_shape=context["in_shape"],
            free_vars=free_variables,
            buf_claim=0,
        )

    @classmethod
    def get_flattern(cls, context, component: nn.Module):
        """处理Flattern模块

        This function ignores Flattern module

        Args:
            context (dict): 编译上下文
            component (nn.Module): Flattern模块

        Returns:
            CompilerToken: CompilerToken
        """

        free_variables = [context["dout"]]
        context["dout"] = ""

        res = f"/** Flatten **/"
        return CompilerToken(
            c_fn=res,
            c_params=None,
            out_shape=cls.get_out_shape(component, context["in_shape"]),
            free_vars=free_variables,
            buf_claim=0,
        )

    @classmethod
    def get_custom(cls, context, component: nn.Module):
        """处理自定义模块
        
        This function add hint for custom modules
        
        Args:
            context (dict): 编译上下文
            component (nn.Module): 自定义模块
        
        Returns:
            CompilerToken: CompilerToken
        """
        layer_id = context["id"]

        free_variables = [context["dout"]]
        context["dout"] = ""

        res = f"/** BEGIN CUSTOM MODULE {layer_id} **/\n\n/**  END CUSTOM MODULE {layer_id}  **/"
        return CompilerToken(
            c_fn=res,
            c_params=None,
            out_shape=cls.get_out_shape(component, context["in_shape"]),
            free_vars=free_variables,
            buf_claim=0,
        )

    @classmethod
    def get_null(cls, context, component: nn.Module):
        """处理需要忽略的模块
        
        This function add hint for custom modules
        
        Args:
            context (dict): 编译上下文
            component (nn.Module): 自定义模块

        Returns:
            CompilerToken: CompilerToken
        """
        component_name = context["name"]

        free_variables = [context["dout"]]
        context["dout"] = ""

        res = f"/** {component_name} **/"
        return CompilerToken(
            c_fn=res,
            c_params=None,
            out_shape=context["in_shape"],
            free_vars=free_variables,
            buf_claim=0,
        )


class Compiler:
    re0 = re.compile("([^\(\s]*)\(.*")
    return_type: str = "pythorch_err_t"

    def __init__(
        self, dtype: str = "f32", app: str = "calc", base_dir: str = "./export"
    ):
        """从Pytorch SequentialModule到C代码的编译器

        Args:
            dtype (str, optional):  应用的数据类型，目前只支持单精度. Defaults to "f32".
            app (str, optional): AI应用，将用于函数名称的前缀. Defaults to "calc".
            base_dir (str, optional): 导出的C函数保存的目录. Defaults to "./export".
        """
        self.dtype = dtype
        self.app = app
        self.base_dir = base_dir

        # 为每一种支持的模块设定一个路由
        self.template_map = {
            "Conv2d": CompilerTemplate.get_conv2d,
            "Linear": CompilerTemplate.get_linear,
            "MaxPool2d": CompilerTemplate.get_maxpool2d,
            "ReLU": CompilerTemplate.get_relu,
            "Flatten": CompilerTemplate.get_flattern,
            "AvgPool2d": CompilerTemplate.get_avgpool2d,
            "Sigmoid": CompilerTemplate.get_sigmoid,
            "Tanh": CompilerTemplate.get_tanh,
            "Dropout": CompilerTemplate.get_null,
            "Module": CompilerTemplate.get_custom,
        }

        # 为数据类型确定映射：f32->float
        self.dtype_mapping = {"f32": "float"}
        # 为了支持AVX/SSE，变量需要对齐
        self.var_attr: str = "__attribute__ ((aligned (32)))"

    def init_context(self):
        """初始化编译上下文

        设定dtype和app属性。这两个属性不会变化

        Returns:
            Dict[str, Any]: 编译上下文
        """
        return {"dtype": self.dtype, "app": self.app}

    @classmethod
    def variable_generator(cls, base: str = "var", index: int = 0):
        """生成一系列下标自增的变量（名称）

        Args:
            base (str, optional): 名称前缀. Defaults to "var".
            index (int, optional): 初始下标. Defaults to 0.

        Yields:
            str: 变量名
        """
        while True:
            yield base + "_" + str(index)
            index += 1

    def export_params(self, c_param_contexts, c_g_buf_size):
        """导出模型权重

        - torch.Tensor作为权重导出为float[]
        - torch.Size作为数组尺寸导出为int[]
        - 增加一个float ${app}_g_buf[] 静态数组作为动态分配内存对替代

        Args:
            c_param_contexts (Dict[str, Union[torch.Tensor, torch.Size]): 模型用到的权重/尺寸，KV键值对
            c_g_buf_size (int): 模型全局缓冲区的尺寸(单位：元素)，可以理解为本来需要在堆区分配c_g_buf_size * sizeof(dtype)内存作为缓冲，现在不使用malloc，就需要在static区手动分配内存这么多内存

        Raises:
            NotImplementedError: 输出的权重类型不合法

        Returns:
            Dict[str, str]: 文件名:文件内容的KV对
        """
        # 写入头文件内容
        data_type = self.dtype_mapping[self.dtype]
        header_buf = ""
        header_buf += "#pragma once\n\n"
        header_buf += f"extern float {self.app}_g_buf[];\n"

        # 写入源文件内容
        src_buf = ""
        src_buf += f'#include "{self.app}_params.h"\n\n'
        src_buf += (
            f"{self.var_attr} float {self.app}_g_buf[{c_g_buf_size}] = " + "{ 0 };\n"
        )

        # 生成c文件
        for name, value in c_param_contexts.items():
            # 输出权重形状数据(整数数组)
            if isinstance(value, torch.Size):
                header_buf += f"extern const int {name}[];" + "\n"
                src_buf += f"const int {name}[]=" + "{"
                for n in value:
                    src_buf += f"{n}, "
                src_buf += "};\n"
            # 输出权重数据(float[])
            elif isinstance(value, torch.Tensor):
                header_buf += f"extern const {data_type} {name}[];" + "\n"
                src_buf += f"{self.var_attr} const {data_type} {name}[]=" + "{"
                for n, v in enumerate(value.flatten()):
                    if n % 10 == 0:
                        src_buf += "\n    "
                    src_buf += "({}){:.8f}, ".format(data_type, v)
                src_buf += "\n};\n"
                pass
            else:
                raise NotImplementedError

        return {f"{self.app}_params.h": header_buf, f"{self.app}_params.c": src_buf}

    def export_function(self, c_fn_contexts, c_var_contexts):
        """导出SequantialModule的对应

        Args:
            c_fn_contexts (List[str]): 每个Module生成的C代码
            c_var_contexts (Dict[str, List[torch.Size]]): 每个变量的大小变化记录

        Returns:
            Dict[str, str]: 文件名:文件内容的KV对
        """
        data_type = self.dtype_mapping[self.dtype]

        # 写入头文件内容
        header_buf = ""
        header_buf += "#pragma once\n\n"
        header_buf += '#include "pythorch/pythorch.h"\n'
        header_buf += (
            f"{self.return_type} {self.app}_fn({data_type}* din, {data_type}* dout);"
            + "\n"
        )

        # 写入源文件内容
        src_buf = ""
        src_buf += '#include "pythorch/pythorch.h"\n'
        src_buf += f'#include "{self.app}_params.h"\n'

        # 计算出每个变量（din, dout）尺寸的历史最大值，按照这个值静态分配变量的内存
        for var_name in c_var_contexts.keys():
            var_size = max(map(lambda x: math.prod(x), c_var_contexts[var_name]))
            src_buf += f"{self.var_attr} {data_type} {var_name}[{var_size}];" + "\n"
        src_buf += "\n"

        # 写入主函数${self.app}_fn，字符串拼接
        src_buf += (
            f"{self.return_type} {self.app}_fn({data_type}* din, {data_type}* dout) "
            + "{\n\n"
        )
        src_buf += f"    {self.return_type} err = PYTHORCH_OK;" + "\n\n"
        for item in c_fn_contexts:
            src_buf += "    " + item + "\n"
        src_buf += f"\n    return err;" + "\n\n}\n"

        return {f"{self.app}_fn.h": header_buf, f"{self.app}_fn.c": src_buf}

    def dump(self, contents: Dict[str, str]):
        """将编译的文件写入磁盘

        Args:
            contents (Dict[str, str]): 文件名:文件内容的KV对
        """
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        for filename, content in contents.items():
            with open(os.path.join(self.base_dir, filename), "w") as f:
                f.write(content)

    def compile(self, seq: nn.Sequential, in_shape: Tuple[int]):
        """编译逻辑

        Args:
            seq (nn.Sequential): Pytorch SequentialModule
            in_shape (Tuple[int]): 输入尺寸

        Returns:
            Dict[str, str]: 文件名:文件内容的KV对
        """
        c_param_contexts: Dict[str, torch.Tensor] = {} # 保存所有的模型权重
        c_fn_contexts: List[Dict[str, Union[str, int]]] = [] # 保存生成的C函数段
        c_var_contexts: Dict[str, List[List[int]]] = {"din": [], "dout": []} # 记录中间变量的分配和尺寸变化
        c_compile_contexts: List[Dict[str, Any]] = [] # 编译上下文
        c_g_buf_size: int = 0 # 追踪临时变量的极大值
        var_generator = self.variable_generator() # 临时变量里那个生成器

        varaibles = deque() # 记录空闲临时变量的队列
        curr_context = self.init_context() # 初始化编译上下文
        curr_context["in_shape"] = list(in_shape) # 设定输出尺寸
        curr_context["din"] = "din" # 入参名称，固定为din，类型为float*

        for layer_id, component in enumerate(seq):
            curr_context["id"] = layer_id  # 给每一个component分配一个独一ID

            if layer_id >= len(seq) - 1:
                curr_context["dout"] = "dout" # 如果是最后一层，出参名称固定为dout
            else:
                while len(varaibles) > 0:
                    curr_context["dout"] = varaibles.pop() # 选择一个已经空闲的变量保存dout
                if "dout" not in curr_context or curr_context["dout"] == "":
                    curr_context["dout"] = next(var_generator)  # 分配Lease一个变量，用掉一个名字
                    if curr_context["dout"] not in c_var_contexts.keys():
                        c_var_contexts[curr_context["dout"]] = [] # 初始化该变量的记录为空

            component_name = self.re0.findall(component.__repr__())[0] # 正则匹配模块的名称
            curr_context["name"] = component_name # 修改Context

            token = self.template_map[component_name](curr_context, component) # 根据路由规则匹配处理函数，得到token

            c_fn_contexts.append(token.c_fn) # 保存token中的C函数体
            c_param_contexts.update(
                **token.c_params
            ) if token.c_params is not None else None # 如果该component产生了权重，记录权重数据
            c_var_contexts[curr_context["din"]].append(token.out_shape) # 跟踪使用的变量尺寸的变化
            c_g_buf_size = int(max(c_g_buf_size, token.buf_claim)) # 如果component领用的临时变量大小超过了全局缓冲大小，增大全局缓冲

            curr_context["in_shape"] = token.out_shape # 更新上下文，为下一个模块准备。下一个模块的输入尺寸是上一个模块的输出尺寸
            varaibles.extend(
                filter(lambda x: x not in ["din", "dout"], token.free_vars) # 将上一个模块使用完的变量放回变量队列
            )
            c_compile_contexts.append(deepcopy(curr_context)) # 跟踪编译上下文的变化
            pass

        del c_var_contexts["din"]
        del c_var_contexts["dout"]

        function_files = self.export_function(c_fn_contexts, c_var_contexts) # 套入模版生成${app}_fn.c/h
        params_files = self.export_params(c_param_contexts, c_g_buf_size) # 套入模版生成${app}_params.c/h
        contents = {**function_files, **params_files}

        self.dump(contents) # 写入磁盘

        return contents # 返回C函数的内容


if __name__ == "__main__":
    # 内建的简单测试
    seq = nn.Sequential(
        nn.Conv2d(1, 32, 5, 1),
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

    comp = Compiler()
    comp.compile(seq, in_shape=(1, 1, 28, 28))
