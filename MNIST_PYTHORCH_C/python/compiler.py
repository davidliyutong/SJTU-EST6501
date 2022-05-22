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


class Template:

    def __init__(self):
        pass

    @classmethod
    def get_out_shape(cls, component, in_shape):
        with torch.no_grad():
            dummy_input = torch.randn(in_shape)
            dummy_output = component(dummy_input)
            out_shape = dummy_output.shape
        return out_shape

    @classmethod
    def get_conv2d(cls, context, component: nn.Module):

        layer_id: str = context['id']
        op: str = 'conv2d' + '_' + context['dtype']  # PYTHORCH_CONV2D_OP
        in_hgt, in_wid = context['in_shape'][2:4]  # PYTHORCH_CONV2D_ID_IN_HGT, PYTHORCH_CONV2D_ID_IN_WID
        # PYTHORCH_CONV2D_ID_WEIGHT
        # PYTHORCH_CONV2D_ID_BIAS
        # PYTHORCH_CONV2D_ID_SHAPE
        dout = context['dout']
        din = context['din']

        state_dict = component.state_dict()
        
        free_variables = [context['din']]
        context['din'] = context['dout']
        context['dout'] = ''

        res = f"{op}({dout}, {din}, {in_hgt}, {in_wid}, {op}_{layer_id}_weight, {op}_{layer_id}_bias, {op}_{layer_id}_shape);"
        return res, \
               {f'{op}_{layer_id}_weight': state_dict['weight'], f'{op}_{layer_id}_bias': state_dict['bias'], f'{op}_{layer_id}_shape': state_dict['weight'].shape}, \
               cls.get_out_shape(component, context['in_shape']), \
               free_variables


    @classmethod
    def get_linear(cls, context, component: nn.Module):
        layer_id: str = context['id']
        op: str = 'linear' + '_' + context['dtype']  # PYTHORCH_LINEAR_OP
        # PYTHORCH_LINEAR_ID_WEIGHT
        # PYTHORCH_LINEAR_ID_BIAS
        # PYTHORCH_LINEAR_ID_SHAPE
        dout = context['dout']
        din = context['din']

        state_dict = component.state_dict()
        
        free_variables = [context['din']]
        context['din'] = context['dout']
        context['dout'] = ''

        res = f"{op}({dout}, {din}, {op}_{layer_id}_weight, {op}_{layer_id}_bias, {op}_{layer_id}_shape);"
        return res, \
               {f'{op}_{layer_id}_weight': state_dict['weight'], f'{op}_{layer_id}_bias': state_dict['bias'], f'{op}_{layer_id}_shape': state_dict['weight'].shape}, \
               cls.get_out_shape(component, context['in_shape']), \
               free_variables

    @classmethod
    def get_maxpool2d(cls, context, component: nn.Module):
        op: str = 'maxpool2d' + '_' + context['dtype']  # PYTHORCH_MAXPOOL2D_OP
        num_c, in_hgt, in_wid = context['in_shape'][1:4]  # PYTHORCH_MAXPOOL2D_ID_IN_HGT, PYTHORCH_MAXPOOL2D_ID_IN_WID, PYTHORCH_MAXPOOL2D_ID_NUM_C
        k_size: int = component.kernel_size  # PYTHORCH_MAXPOOL2D_ID_K_SIZE 24
        # PYTHORCH_LINEAR_ID_WEIGHT
        # PYTHORCH_LINEAR_ID_BIAS
        # PYTHORCH_LINEAR_ID_SHAPE
        dout = context['dout']
        din = context['din']

        free_variables = [context['din']]
        context['din'] = context['dout']
        context['dout'] = ''

        res = f"{op}({dout}, {din}, {in_hgt}, {in_wid}, {num_c}, {k_size});"
        return res, \
               None, \
               cls.get_out_shape(component, context['in_shape']), \
               free_variables

    @classmethod
    def get_avgpool2d(cls, context, component: nn.Module):
        op: str = 'avgpool2d' + '_' + context['dtype']  # PYTHORCH_AVGPOOL2D_OP
        num_c, in_hgt, in_wid = context['in_shape'][1:4]  # PYTHORCH_AVGPOOL2D_ID_IN_HGT, PYTHORCH_AVGPOOL2D_ID_IN_WID, PYTHORCH_AVGPOOL2D_ID_NUM_C
        k_size: int = component.kernel_size  # PYTHORCH_AVGPOOL2D_ID_K_SIZE 24
        # PYTHORCH_LINEAR_ID_WEIGHT
        # PYTHORCH_LINEAR_ID_BIAS
        # PYTHORCH_LINEAR_ID_SHAPE
        dout = context['dout']
        din = context['din']

        free_variables = [context['din']]
        context['din'] = context['dout']
        context['dout'] = ''

        res = f"{op}({dout}, {din}, {in_hgt}, {in_wid}, {num_c}, {k_size});"
        return res, \
               None, \
               cls.get_out_shape(component, context['in_shape']), \
               free_variables

    @classmethod
    def get_relu(cls, context, component: nn.Module):
        op = 'relu' + '_' + context['dtype']  # PYTHORCH_RELU_OP
        size = math.prod(context['in_shape'])  # PYTHORCH_RELU_SIZE
        dout = context['dout']
        din = context['din']

        free_variables = [context['din']]
        context['din'] = context['dout']
        context['dout'] = ''

        res = f"{op}({dout}, {din}, {size});"
        return res, \
               None, \
               context['in_shape'], \
               free_variables

    @classmethod
    def get_sigmoid(cls, context, component: nn.Module):
        op = 'sigmoid' + '_' + context['dtype']  # PYTHORCH_SIGMOID_OP
        size = math.prod(context['in_shape'])  # PYTHORCH_SIGMOID_SIZE
        dout = context['dout']
        din = context['din']

        free_variables = [context['din']]
        context['din'] = context['dout']
        context['dout'] = ''

        res = f"{op}({dout}, {din}, {size});"
        return res, \
               None, \
               context['in_shape'], \
               free_variables

    @classmethod
    def get_tanh(cls, context, component: nn.Module):
        op = 'tanh' + '_' + context['dtype']  # PYTHORCH_TANH_OP
        size = math.prod(context['in_shape'])  # PYTHORCH_TANH_SIZE
        dout = context['dout']
        din = context['din']

        free_variables = [context['din']]
        context['din'] = context['dout']
        context['dout'] = ''

        res = f"{op}({dout}, {din}, {size});"
        return res, \
               None, \
               context['in_shape'], \
               free_variables

    @classmethod
    def get_flattern(cls, context, component: nn.Module):
        """Convert Flattern
        
        This function ignores Flattern module

        Args:
            context (dict): compiler context
            component (nn.Module): nn.Module

        Returns:
            Tuple: "", None, out_shape
        """

        free_variables = [context['dout']]
        context['dout'] = ''

        res = f"/** Flatten **/"
        return res, \
               None, \
               cls.get_out_shape(component, context['in_shape']), \
               free_variables

    @classmethod
    def get_custom(cls, context, component: nn.Module):
        """Convert custom modules
        
        This function add hint for custom modules
        
        Args:
            context (dict): compiler context
            component (nn.Module): nn.Module

        Returns:
            Tuple: "/**...**/", None, out_shape
        """
        layer_id = context['id']
        
        free_variables = [context['dout']]
        context['dout'] = ''

        res = f"/** BEGIN CUSTOM MODULE {layer_id} **/\n\n/**  END CUSTOM MODULE {layer_id}  **/"
        return res, \
               None, \
               cls.get_out_shape(component, context['in_shape']), \
               free_variables

    @classmethod
    def get_null(cls, context, component: nn.Module):
        """Ignore  modules
        
        This function add hint for custom modules
        
        Args:
            context (dict): compiler context
            component (nn.Module): nn.Module

        Returns:
            Tuple: "", None, out_shape=in_shape
        """
        component_name = context['name']

        free_variables = [context['dout']]
        context['dout'] = ''

        res = f"/** {component_name} **/"
        return res, \
               None, \
               context['in_shape'], \
               free_variables


class Compiler:
    re0 = re.compile('([^\(\s]*)\(.*')
    return_type: str = 'pythorch_err_t'

    def __init__(self, dtype: str = 'f32', name: str = 'calc', base_dir: str = "./export"):
        self.dtype = dtype
        self.name = name
        self.base_dir = base_dir
        self.template_map = {
            'Conv2d': Template.get_conv2d,
            'Linear': Template.get_linear,
            'MaxPool2d': Template.get_maxpool2d,
            'ReLU': Template.get_relu,
            'Flatten': Template.get_flattern,
            'AvgPool2d': Template.get_avgpool2d,
            'Sigmoid': Template.get_sigmoid,
            'Tanh': Template.get_tanh,
            'Dropout': Template.get_null,
            'Module': Template.get_custom,
        }
        self.dtype_mapping = {'f32': 'float'}

    def init_context(self):
        return {'dtype': self.dtype}

    @classmethod
    def variable_generator(cls, base: str = 'var', index: int = 0):
        while True:
            yield base + '_' + str(index)
            index += 1

    def export_params(self, c_param_contexts):
        data_type = self.dtype_mapping[self.dtype]
        header_buf = ""
        header_buf += "#pragma once\n\n"

        src_buf = ""
        src_buf += f"#include \"{self.name}_params.h\"\n\n"

        # 生成c文件
        for name, value in c_param_contexts.items():
            # 输出权重形状数据(整数数组)

            if isinstance(value, torch.Size):
                header_buf += f"extern const int {name}[];" + "\n"
                src_buf += f"const int {name}[]=" + "{"
                for n in value:
                    src_buf += f"{n}, "
                src_buf += "};\n"
            elif isinstance(value, torch.Tensor):
                header_buf += f"extern const {data_type} {name}[];" + "\n"
                src_buf += f"const {data_type} {name}[]=" + "{"
                for n, v in enumerate(value.flatten()):
                    if n % 10 == 0: src_buf += "\n    "
                    src_buf += "({}){:.8f}, ".format(data_type, v)
                src_buf += "\n};\n"
                pass
            else:
                raise NotImplementedError

        return {f"{self.name}_params.h": header_buf, f"{self.name}_params.c": src_buf}

    def export_function(self, c_fn_contexts, c_var_contexts):
        data_type = self.dtype_mapping[self.dtype]

        # dump params
        header_buf = ""
        header_buf += "#pragma once\n\n"
        header_buf += "#include \"pythorch/modules.h\"\n"
        header_buf += f"{self.return_type} {self.name}_fn({data_type}* din, {data_type}* dout);" + "\n"

        # dump function c
        src_buf = ""
        src_buf += "#include \"pythorch/modules.h\"\n"
        src_buf += f"#include \"{self.name}_params.h\"\n"

        for var_name in c_var_contexts.keys():
            var_size = max(map(lambda x: math.prod(x), c_var_contexts[var_name]))
            src_buf += f"{data_type} {var_name}[{var_size}];" + "\n"
        src_buf += "\n"
        src_buf += f"{self.return_type} {self.name}_fn({data_type}* din, {data_type}* dout) " + "{\n\n"
        src_buf += f"    {self.return_type} err = PYTHORCH_OK;" + "\n\n"
        for item in c_fn_contexts:
            src_buf += "    " + item + "\n"
        src_buf += f"\n    return err;" + "\n\n}\n"

        return {f"{self.name}_fn.h": header_buf, f"{self.name}_fn.c": src_buf}

    def dump(self, contents: Dict[str, str]):
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        for filename, content in contents.items():
            with open(os.path.join(self.base_dir, filename), "w") as f:
                f.write(content)

    def compile(self, seq: nn.Sequential, in_shape: Tuple[int]):
        c_param_contexts: Dict[str, torch.Tensor] = {}
        c_fn_contexts: List[Dict[str, Union[str, int]]] = []
        c_var_contexts: Dict[str, List[List[int]]] = {'din': [], 'dout': []}
        c_compile_contexts: List[Dict[str, Any]] = []
        var_generator = self.variable_generator()

        varaibles = deque()
        curr_context = self.init_context()
        curr_context['in_shape'] = list(in_shape)
        curr_context['din'] = 'din'

        for layer_id, component in enumerate(seq):
            curr_context['id'] = layer_id  # Assign id

            if layer_id >= len(seq) - 1:
                curr_context['dout'] = 'dout'
            else:
                while len(varaibles) > 0:
                    curr_context['dout'] = varaibles.pop()
                if 'dout' not in curr_context or curr_context['dout'] == '':
                    curr_context['dout'] = next(var_generator)  # Lease var
                    if curr_context['dout'] not in c_var_contexts.keys():
                        c_var_contexts[curr_context['dout']] = []

            component_name = self.re0.findall(component.__repr__())[0]
            curr_context['name'] = component_name

            c_fn, c_param, out_shape, free_vars = self.template_map[component_name](curr_context, component)

            c_fn_contexts.append(c_fn)
            c_param_contexts.update(**c_param) if c_param is not None else None
            c_var_contexts[curr_context['din']].append(out_shape)

            curr_context['in_shape'] = out_shape
            varaibles.extend(filter(lambda x: x not in ['din', 'dout'], free_vars))
            c_compile_contexts.append(deepcopy(curr_context))
            pass

        del c_var_contexts['din']
        del c_var_contexts['dout']

        function_files = self.export_function(c_fn_contexts, c_var_contexts)
        params_files = self.export_params(c_param_contexts)
        contents = {**function_files, **params_files}

        self.dump(contents)

        return contents


if __name__ == '__main__':
    # Bundled test
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