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

# BatchNormld类的定义
class BatchNorm1d:
    def __init__(self,dim,eps=1e-5,momentum=0.1):
       self.momentum=momentum                       # 更新运行统计量时，新一批的占比
       self.eps=eps                                 # 防止除以0
       self.training=True                           # 标记是否使用训练模式
       self.gamma=torch.ones((1,dim))               # 可学习的缩放参数
       self.beta=torch.zeros((1,dim))               # 可学习的平移参数
       self.running_mean=torch.zeros((1,dim))       # 保存运行均值
       self.running_var=torch.ones((1,dim))         # 保存运行方差


    def __call__(self, x):
        if self.training is True:
            # 每个神经元各自的均值、方差：[1，200]
            bnmean=x.mean(dim=0,keepdim=True)
            bnvar=x.var(dim=0,keepdim=True,unbiased=False)

            # 标准化后的结果
            hnorm=(x-bnmean)/torch.sqrt(bnvar+self.eps)     # +1e-5是一个很小的数，防止分母为0
            self.out=self.gamma*hnorm+self.beta

            # 更新累计的统计量：
            # moment的取值表示本次统计量的占比，例如moment=0.1，新统计量=90%旧统计量+10%新本批次统计量，避免统计结果完全由某一批次的样本决定
            with torch.no_grad():
                self.running_mean=((1-self.momentum)*self.running_mean+self.momentum*bnmean)
                self.running_var=((1-self.momentum)*self.running_var+self.momentum*bnvar) 

            return self.out 

        else:
            hnorm=(x-self.running_mean)/torch.sqrt(self.running_var+self.eps)
            self.out=self.gamma*hnorm+self.beta
            return self.out
        
    def parameters(self):
        return [self.gamma,self.beta]
