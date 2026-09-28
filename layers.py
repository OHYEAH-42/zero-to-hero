import torch

# 全连接层的定义
class Linear:
    # 初始化全连接层，包含输入数量，输出数量，是否偏置；权重，偏置
    def __init__(self,fan_in,fan_out,bias=True):
        self.weight=torch.randn((fan_in,fan_out))/fan_in**0.5                # 权重

        if bias:                                                            # 偏置
            self.bias=torch.zeros(fan_out)
        else:
            self.bias=None

    # 返回该层的输出，也就是x@w+b
    def __call__(self, x):                          
        self.out=x@self.weight
        if self.bias is not None:
            self.out+=self.bias
        return self.out
    
    # 返回该层的参数
    def parameters(self):
        params=[self.weight]
        if self.bias is not None:
            params+=[self.bias]
        return params

# Tanh类的定义
class Tanh:
    # 计算tanh，返回计算值
    def __call__(self,x):
        self.out=torch.tanh(x)
        return self.out

    # 返回该层参数
    def parameters(self):
        return []