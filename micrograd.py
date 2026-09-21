# micrograd.py的实现，包含Value类和反向传播函数
import math
import random

# value类，data为值，grad为梯度，op为操作符,_backward为反向传播函数，children为子节点集合
class Value:
    def __init__(self, data,op=None,children=()):
        self.data = data
        self.grad =0 
        self.op = op
        self._prev=set(children)
        self._backward = lambda: None

    
    # 重载加法和乘法运算符，返回一个新的Value对象，并定义反向传播函数
    # 注意是定义反向传播函数_backward，而不是直接调用反向传播函数 
    def __add__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data+other.data,"+",(self,other))
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data*other.data,"*",(self,other))
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    # 重载幂运算符，返回一个新的Value对象，并定义反向传播函数
    def __pow__(self,other):
       out = Value(self.data**other,f"**{other}",(self,))
       def _backward():
         self.grad += other*(self.data)**(other-1)*out.grad
       out._backward = _backward 
       return out

    # 重载反向加法和乘法运算符，返回一个新的Value对象，并定义反向传播函数
    def __radd__(self,other):
        return self + other

    def __rmul__(self,other):
        return self * other

    # 重载正向和反向减法和除法运算符，返回一个新的Value对象，并定义反向传播函数
    def __sub__(self,other):
        return self + (-1*other)

    def __rsub__(self,other):
        return other + (-1*self)

    def __truediv__(self,other):
        return self*other**-1

    def __rtruediv__(self,other):
        return other*self**-1

    # 重载负号运算符，返回一个新的Value对象，并定义反向传播函数
    def __neg__(self):
        return self * -1

    # 反向传播函数，使用深度优先搜索构建拓扑排序，然后从输出节点开始反向传播梯度
    def backward(self):
        topo = []
        visited = set()

        # DFS构建拓扑排序
        def build_topo(v):
           if v not in visited:
               visited.add(v)
               for child in v._prev:
                   build_topo(child)
               topo.append(v) 
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    # 重载__repr__方法，返回Value对象的字符串表示
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    # 定义tanh函数，返回一个新的Value对象，并定义反向传播函数
    def tanh(self):
        x =self.data
        t = math.tanh(x)
        out = Value(t,"tanh",(self,))
        def _backward():
            self.grad+= (1-t**2)*out.grad
        out._backward = _backward
        return out
    
# 定义Module类，包含parameters方法和zero_grad方法，parameters方法返回模型的参数列表，zero_grad方法将所有参数的梯度清零
class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

# 定义Neuron类，包含权重w和偏置b，初始化时随机生成权重和偏置
class Neuron(Module):
    def __init__(self,nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)

    def __call__(self,x):
        act = sum((wi*xi for wi,xi in zip(self.w,x)),self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

# 定义Layer类，包含多个Neuron对象，初始化时生成指定数量的Neuron对象
class Layer(Module):
    def __init__(self,nin,nout):
        self.neurons=[Neuron(nin) for _ in range(nout)]

    def __call__(self,x):
        out=[n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        params = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params

# 定义MLP类，包含多个Layer对象，初始化时生成指定数量的Layer对象
# 多层感知机（MLP）是由多个神经元组成的前馈神经网络，每一层的输出作为下一层的输入，最终输出一个结果。MLP可以用于分类、回归等任务。
class MLP(Module):
    def __init__(self,nin,nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i],sz[i+1]) for i in range(len(nouts))]

    def __call__(self,x):
        for layer in self.layers:
            x=layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

if __name__ == "__main__":
    model = MLP(3, [4, 4, 1])

    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]

    ys = [1.0, -1.0, -1.0, 1.0]


    for step in range(100):
    # 1. 前向传播
        predictions = [model(x) for x in xs]

        # 2. 计算损失
        loss = sum(
            (prediction - target) ** 2
            for prediction, target in zip(predictions, ys)
        )

        # 3. 清空上一轮梯度
        model.zero_grad()

        # 4. 反向传播
        loss.backward()

        # 5. 使用梯度下降更新所有参数
        learning_rate = 0.05

        for parameter in model.parameters():
            parameter.data -= learning_rate * parameter.grad

        # 6. 每十轮显示一次
        if step % 10 == 0:
            print(step, loss.data)
            
print("predictions:")
for prediction, target in zip(predictions, ys):
    print(prediction.data, "target:", target)
