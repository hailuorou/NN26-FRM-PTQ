import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.quantizer import UniformAffineQuantizer
from quantize.act_quantizer import UniformActQuantizer




class QuantLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Linear,
        wbits=4,
        abits=16,
        group_size=64,
        use_act_quant=False
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_parameter('weight',org_module.weight) # trainable
        if org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = use_act_quant
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(wbits, group_size, weight=org_module.weight)
        if self.use_act_quant:
            self.act_quantizer = UniformActQuantizer(abits, group_size)
            print("insert activation quantizer")
        self.use_temporary_parameter = False

    
    
    def forward(self, input: torch.Tensor):
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        if self.use_act_quant:
            input = self.act_quantizer(input)
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)


        return out

    def set_quant_state(self, weight_quant: bool = False, activation_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = activation_quant




