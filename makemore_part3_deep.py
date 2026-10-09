import random
import torch
from layers import Linear, BatchNorm1d, Tanh
import torch.nn.functional as F

# 读取名字并建立字符编号
words = open("names.txt", encoding="utf-8").read().splitlines()
chars = sorted(set("".join(words)))
stoi = {ch: i + 1 for i, ch in enumerate(chars)}
stoi["."] = 0

block_size = 3

def build_dataset(name_list):
    X, Y = [], []
    for word in name_list:
        context = [0] * block_size
        for ch in word + ".":
            index = stoi[ch]
            X.append(context)
            Y.append(index)
            context = context[1:] + [index]
    return torch.tensor(X), torch.tensor(Y)

# 与之前的模型使用相同的数据划分
shuffled_words = words.copy()
random.Random(42).shuffle(shuffled_words)
n1 = int(0.8 * len(shuffled_words))
n2 = int(0.9 * len(shuffled_words))

Xtr, Ytr = build_dataset(shuffled_words[:n1])
Xdev, Ydev = build_dataset(shuffled_words[n1:n2])
Xte, Yte = build_dataset(shuffled_words[n2:])

# 字符嵌入表
g = torch.Generator().manual_seed(2147483647)
embedding_dim = 30
C = torch.rand((27, embedding_dim), generator=g)

# 创建线性层
# 给Linear的随机权重设定种子
torch.manual_seed(2147483647)

hidden_size=200
layers=[]
fan_in=block_size*embedding_dim

for _ in range(5):
    layers.extend([Linear(fan_in,hidden_size,bias=False),BatchNorm1d(hidden_size),Tanh(),])
    fan_in=hidden_size
layers.append(Linear(fan_in,27))

parameters=[C]

for layer in layers:
    parameters.extend(layer.parameters())

for p in parameters:
    p.requires_grad_()

print(sum(p.numel()for p in parameters))

# 训练 
batch_size=32
for step in range(20000):
    learning_rate=0.1 if step<10000 else 0.01
    # 前向传播
    ix=torch.randint(0,Xtr.shape[0],(batch_size,),generator=g)
    x=C[Xtr[ix]].view(batch_size,-1)
    for layer in layers:
        x=layer(x)
        # Tanh的输出保留梯度
        if step==0 and isinstance(layer,Tanh):
            layer.out.retain_grad()

    loss=F.cross_entropy(x,Ytr[ix])

    # 反向传播
    for p in parameters:
        p.grad=None

    loss.backward()

    # 查看初次数据生成是否合理
    if step==0:
        for i, layer in enumerate(layers):
            if isinstance(layer, Tanh):
                print(
                    "层", i,
                    "输出标准差", layer.out.std().item(),
                    "饱和比例", (layer.out.abs() > 0.97).float().mean().item(),
                    "梯度标准差", layer.out.grad.std().item(),
                )

    if step==0:
        for i, layer in enumerate(layers):
            if isinstance(layer, Linear):
                ratio = (
                    (learning_rate * layer.weight.grad).std()
                    / layer.weight.std()
                )
                print("Linear 层", i, "更新/权重：", ratio.item())

    with torch.no_grad():
        for p in parameters:
            p-=learning_rate*p.grad

    if (step+1)%1000==0:
        print(step+1,loss.item())
# BatchNorm 切换到预测模式：使用运行均值和运行方差
for layer in layers:
    if isinstance(layer, BatchNorm1d):
        layer.training = False

def evaluate(X, Y):
    with torch.no_grad():
        x = C[X].view(X.shape[0], -1)
        for layer in layers:
            x = layer(x)
        return F.cross_entropy(x, Y).item()

print("验证集 loss:", evaluate(Xdev, Ydev))
print("测试集 loss:", evaluate(Xte, Yte))

# 随机生成名字
itos = {i: ch for ch, i in stoi.items()}
sample_g = torch.Generator().manual_seed(42)

with torch.no_grad():
    for _ in range(10):  # 生成 10 个名字
        context = [0] * block_size  # 从 "..." 开始
        output = []

        while True:
            x = C[torch.tensor([context])].view(1, -1)  # [1, 90]

            for layer in layers:
                x = layer(x)  # 最终得到 [1, 27] 的 logits

            probabilities = F.softmax(x, dim=1)
            next_index = torch.multinomial(probabilities, num_samples=1, generator=sample_g).item()

            if next_index == 0:  # 生成了结束符 "."
                break

            output.append(itos[next_index])
            context = context[1:] + [next_index]

        print("".join(output))