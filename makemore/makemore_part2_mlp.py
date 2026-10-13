import random
import torch
import torch.nn.functional as F

# ------读取数据------
words=open("names.txt",encoding="utf-8").read().splitlines()

chs=sorted(set("".join(words)))
stoi={ch:i+1 for i,ch in enumerate(chs)}
stoi["."]=0
itos={i:ch for ch, i in stoi.items()}

# ------数据集创建------

# 设置上下文长度
block_size=3

# 数据集生成函数
def build_dataset(name_list):
    X,Y=[],[]

    # 读取上下文以及准取值
    for word in name_list:
        context=[0]*block_size

        for ch in word+".":
            ix=stoi[ch]
            X.append(context)
            Y.append(ix)
            context=context[1:]+[ix]

    return torch.tensor(X),torch.tensor(Y)

# 创建数据集的副本，并随机打乱
shuffled_words=words.copy()
random.Random(42).shuffle(shuffled_words)

# 设定数据集的分界线
n1=int(0.8*len(shuffled_words))
n2=int(0.9*len(shuffled_words))

Xtr,Ytr=build_dataset(shuffled_words[:n1])          # 训练集，占总数据的80%
Xdev,Ydev=build_dataset(shuffled_words[n1:n2])      # 验证集，占总数据的10%
Xte,Yte=build_dataset(shuffled_words[n2:])          # 测试集，占总数据的10%

# ------训练参数构建------

g =torch.Generator().manual_seed(2147483647)
# 27个字符，每个字符用embedding_dim个可学习的数学表示
embedding_dim=30
C = torch.rand((27,embedding_dim),generator=g)

# 隐藏层有hidden_size个神经元，每个神经元有embedding_dim个输入
hidden_size=200
W1=torch.randn((block_size*embedding_dim,hidden_size),generator=g)
b1=torch.zeros(hidden_size)

# hidden_size个隐藏层输出-》27个字符的评分
W2=torch.randn((hidden_size,27),generator=g)
b2=torch.zeros(27)

# 将五组可训练学习的参数统合
parameters=[C,W1,b1,W2,b2]
for p in parameters:
    p.requires_grad_()

# ------训练------

# 选取的训练数据的数量
batch_size=32

# 计算loss函数
def evaluate(X_eval,Y_eval):
    with torch.no_grad():
        emb=C[X_eval]
        h=torch.tanh(emb.view(X_eval.shape[0],block_size*embedding_dim)@W1+b1)
        logits=h@W2+b2
        return F.cross_entropy(logits,Y_eval).item()

# 使用batch_size个样本进行训练
for step in range(200000):
    # 学习率设置：
    learning_rate=0.1 if step<100000 else 0.01
    # 随机选取32条训练数据
    ix =torch.randint(0,Xtr.shape[0],(batch_size,),generator=g)

    # 前向传播：每一轮都要用当前最新的参数重新计算
    # 用C和X构建出嵌入表,shape应该为[batch_size行数，X一行的数据数量block_size,C中一个字符的数字表示个数]
    emb=C[Xtr[ix]]
    emb_flat=emb.view(batch_size,block_size*embedding_dim)
    h=torch.tanh(emb_flat@W1+b1) 
    logits=h@W2+b2
    loss=F.cross_entropy(logits,Ytr[ix])

    # 反向传播
    for p in parameters:
        p.grad=None
    loss.backward()

    with torch.no_grad():
        for p in parameters:
            p-=learning_rate*p.grad

    if (step+1)%10000==0:
        print(step+1,"batch loss: ",loss.item(),"验证集 loss:",evaluate(Xdev,Ydev))

print("训练集 loss:",evaluate(Xtr,Ytr))
print("验证集 loss:",evaluate(Xdev,Ydev))
print("测试集 loss:",evaluate(Xte,Yte))

# ------利用训练好的参数生成名字------
sample_genrator=torch.Generator().manual_seed(42)

with torch.no_grad():
    for _ in range(10):
        context=[0]*block_size
        output=[]

        for _ in range(30):
            x=torch.tensor([context])
            emb=C[x]
            h=torch.tanh(emb.view(1,block_size*embedding_dim)@W1+b1)
            logits=h@W2+b2

            prob=F.softmax(logits,dim=1)
            next_index=torch.multinomial(prob[0],num_samples=1,generator=sample_genrator).item()

            if next_index==0:
                break

            output.append(itos[next_index])
            context=context[1:]+[next_index]

        print("".join(output))