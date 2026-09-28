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
fan_in=block_size*embedding_dim                     # 每个神经元的扇入数/输入数量
gain=5/3                                            # tanh的增益系数
W1=torch.randn((block_size*embedding_dim,hidden_size),generator=g)*gain/fan_in**0.5
b1=torch.zeros(hidden_size)

# hidden_size个隐藏层输出-》27个字符的评分
W2=torch.randn((hidden_size,27),generator=g)*0.01
b2=torch.zeros(27)

bngain=torch.ones((1,hidden_size))                  # gamma,初始化为1
bnbias=torch.zeros((1,hidden_size))                 # beta，初始化为0

# 在训练时，方差和均值都是用当前样本计算的。但是在生成名字时，可能只有一个样本，这个时候就需要使用训练时的累计量
bnmean_running = torch.zeros((1, hidden_size))      # 运行均值，根据训练时多个batch持续更新的均值估计
bnvar_running = torch.ones((1, hidden_size))        # 运行方差，根据训练时多个batch持续更新的方差估计
momentum = 0.1                                      # 每次更新以上两种数据时，当前批次数据的占比

parameters=[C,W1,b1,W2,b2,bngain,bnbias]            # 参数列表
for p in parameters:
    p.requires_grad_()

# 计算loss函数
def evaluate(X_eval,Y_eval):
    with torch.no_grad():
        emb=C[X_eval]
        emb_flat=emb.view(X_eval.shape[0],-1)

        hprebn=emb_flat@W1+b1
        hnorm=(hprebn-bnmean_running)/torch.sqrt(bnvar_running+1e-5)     
        hpreact=bngain*hnorm+bnbias
        h=torch.tanh(hpreact)

        logits=h@W2+b2
        return F.cross_entropy(logits,Y_eval).item()

batch_size=32

# def normalize(x, mean, var):
#     return bngain * (x - mean) / torch.sqrt(var + 1e-5) + bnbias


# with torch.no_grad():
#     # 取 32 个样本，计算进入 BN 前的数据
#     ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
#     emb = C[Xtr[ix]]
#     x = emb.view(32, -1) @ W1 + b1

#     # 取出其中第一个样本，保持二维形状 [1, 200]
#     single = x[:1]

#     # 为这个小实验准备固定的统计量
#     saved_mean = x.mean(dim=0, keepdim=True)
#     saved_var = x.var(dim=0, keepdim=True, unbiased=False)

#     # 训练方式：每次根据当前输入重新计算统计量
#     train_batch = normalize(x, saved_mean, saved_var)
#     train_single = normalize(
#         single,
#         single.mean(dim=0, keepdim=True),
#         single.var(dim=0, keepdim=True, unbiased=False),
#     )

#     # 预测方式：无论输入多少个样本，都使用同一套统计量
#     eval_batch = normalize(x, saved_mean, saved_var)
#     eval_single = normalize(single, saved_mean, saved_var)

#     print("训练方式，一起输入：", train_batch[0, :5])
#     print("训练方式，单独输入：", train_single[0, :5])

#     print("预测方式，一起输入：", eval_batch[0, :5])
#     print("预测方式，单独输入：", eval_single[0, :5])

# # 临时结束程序，只运行这次实验
# raise SystemExit

# ------ 训练 ------
for step in range(200000):
    # 学习率设置：
    learning_rate=0.1 if step<100000 else 0.01
    # 随机选取32条训练数据
    ix =torch.randint(0,Xtr.shape[0],(batch_size,),generator=g)

    # ------ 前向传播：每一轮都要用当前最新的参数重新计算 ------
    # 用C和X构建出嵌入表,shape应该为[batch_size行数，X一行的数据数量block_size,C中一个字符的数字表示个数]
    emb=C[Xtr[ix]]
    emb_flat=emb.view(batch_size,block_size*embedding_dim)

    # ------ 批归一化 ------
    hprebn=emb_flat@W1+b1                               # 标准化之前的数据

    # 每个神经元各自的均值、方差：[1，200]
    bnmean=hprebn.mean(dim=0,keepdim=True)
    bnvar=hprebn.var(dim=0,keepdim=True,unbiased=False)

    # 标准化后的结果
    hnorm=(hprebn-bnmean)/torch.sqrt(bnvar+1e-5)     # +1e-5是一个很小的数，防止分母为0
    hpreact=bngain*hnorm+bnbias
    h=torch.tanh(hpreact)

    # 更新累计的统计量：
    # moment的取值表示本次统计量的占比，例如moment=0.1，新统计量=90%旧统计量+10%新本批次统计量，避免统计结果完全由某一批次的样本决定
    with torch.no_grad():
        batch_var=hprebn.var(dim=0,keepdim=True,unbiased=False)

        bnmean_running=((1-momentum)*bnmean_running+momentum*bnmean)
        bnvar_running=((1-momentum)*bnvar_running+momentum*bnvar) 
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
print("logits 标准差: ",logits.std().item())
print("tanh 输入标准差：", hpreact.std().item())

saturation=(h.abs()>0.97).float().mean()
print("tanh 饱和比例：",saturation.item())

print("BN 后各列均值：", hpreact.mean(dim=0)[:5])
print("BN 后各列标准差：",hpreact.std(dim=0, unbiased=False)[:5],)

# ------利用训练好的参数生成名字------
sample_genrator=torch.Generator().manual_seed(42)

with torch.no_grad():
    for _ in range(10):
        context=[0]*block_size
        output=[]

        for _ in range(30):
            x=torch.tensor([context])
            emb=C[x]
            emb_flat=emb.view(x.shape[0],-1)

            hprebn=emb_flat@W1+b1
            hnorm=(hprebn-bnmean_running)/torch.sqrt(bnvar_running+1e-5)
            hpreact=bngain*hnorm+bnbias
            h=torch.tanh(hpreact)
            
            logits=h@W2+b2

            prob=F.softmax(logits,dim=1)
            next_index=torch.multinomial(prob[0],num_samples=1,generator=sample_genrator).item()

            if next_index==0:
                break

            output.append(itos[next_index])
            context=context[1:]+[next_index]

        print("".join(output))