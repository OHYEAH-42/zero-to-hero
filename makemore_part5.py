import torch
import random

from layers import(
    Embedding,
    FlattenConsecutive,
    Linear,
    Tanh,
    BatchNorm1d,
    Sequential
)

torch.manual_seed(42)

# -------------------------------------------------------------读取名字并建立字符编号-----------------------------------------------------------------------------
words = open("names.txt", encoding="utf-8").read().splitlines()
chars = sorted(set("".join(words)))
stoi = {ch: i + 1 for i, ch in enumerate(chars)}
stoi["."] = 0

block_size = 8

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

# -------------------------------------------------------------构建模型-----------------------------------------------------------------------------
vocab_size=len(stoi)                           # 字符表大小
embedding_dim=10                        # 嵌入向量的维度
hidden_size=200                         # 隐藏层的大小

model=Sequential([
    Embedding(vocab_size,embedding_dim),         # 27个字符，每个字符用embedding_dim个可学习的数学表示

    # 第一组：8->4
    FlattenConsecutive(2),                          # 将相邻的两个位置的特征展平为一个向量
    Linear(embedding_dim*2,hidden_size,bias=False), # 全连接层，输入为embedding_dim*2，输出为hidden_size
    BatchNorm1d(hidden_size),                       # 批归一化层，输入为hidden_size
    Tanh(),                                         # 激活函数，输入为hidden_size    

    # 第二组：4->2
    FlattenConsecutive(2),      
    Linear(hidden_size*2,hidden_size,bias=False), 
    BatchNorm1d(hidden_size),
    Tanh(),

    # 第三组：2->1
    FlattenConsecutive(2),
    Linear(hidden_size*2,hidden_size,bias=False),
    BatchNorm1d(hidden_size),
    Tanh(),

    Linear(hidden_size, vocab_size)                 # 输出层，输入为hidden_size，输出为vocab_size
])

# -------------------------------------------------------------初始化模型参数-----------------------------------------------------------------------------
with torch.no_grad():
    model.layers[-1].weight *= 0.1

parameters=model.parameters()
for p in parameters:
    p.requires_grad_()

print("参数数量：",sum(p.numel() for p in parameters))

# -------------------------------------------------------------训练-----------------------------------------------------------------------------
max_steps=20000                                     # 训练步数
batch_size=32                                       # 每个批次的样本数
lossi=[]

for step in range(max_steps):
    # 采样一个批次
    ix=torch.randint(0,Xtr.shape[0],(batch_size,))
    Xb,Yb=Xtr[ix],Ytr[ix]

    # 前向传播
    logits=model(Xb)
    loss=torch.nn.functional.cross_entropy(logits,Yb)

    # 反向传播
    for p in parameters:
        p.grad=None
    loss.backward()

    # 更新参数
    lr=0.1 if step<10000 else 0.01
    with torch.no_grad():
        for p in parameters:
            p -= lr * p.grad

    lossi.append(loss.log10().item())

    if (step+1)%1000==0:
        print(step+1,"batch loss:",loss.item())

# -------------------------------------------------------------评估-----------------------------------------------------------------------------
# 评估时使用训练期间累计的均值和方差
for layer in model.layers:
    if isinstance(layer,BatchNorm1d):
        layer.training=False

# 评估函数
def evaluate(X,Y):
    total_loss=0.0                                  # 累计损失
    eval_batch_size=1000                            # 每次评估的批次大小

    with torch.no_grad():
        for start in range(0,X.shape[0],eval_batch_size):
            Xb=X[start:start+eval_batch_size]
            Yb=Y[start:start+eval_batch_size]
            logits=model(Xb)

            # 累加这一批样本所有的损失
            loss=torch.nn.functional.cross_entropy(logits,Yb,reduction="sum")
            total_loss+=loss.item()

    # 返回平均损失
    return total_loss/X.shape[0]

print("训练集 loss:",evaluate(Xtr,Ytr))
print("验证集 loss:",evaluate(Xdev,Ydev))
print("测试集 loss:", evaluate(Xte, Yte))

# -------------------------------------------------------------生成名字-----------------------------------------------------------------------------

itos = {i: ch for ch, i in stoi.items()}
sample_g = torch.Generator().manual_seed(42)

with torch.no_grad():
    for _ in range(10):  # 生成 10 个名字
        context = [0] * block_size  # 从 "........" 开始
        output = []

        for _ in range(30):  # 最多生成 30 个字符
            x=model(torch.tensor([context]))
            probabilities=torch.nn.functional.softmax(x,dim=1)
            next_index=torch.multinomial(probabilities,num_samples=1,generator=sample_g).item()

            if next_index == 0:  # 生成了结束符 "."
                break

            output.append(itos[next_index])
            context = context[1:] + [next_index]

        print("".join(output))