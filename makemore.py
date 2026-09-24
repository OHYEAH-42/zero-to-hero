# makemore的实现
import torch
import torch.nn.functional as F

# 读取名字数据
words=open('names.txt').read().splitlines()

# print(words[:10])
# print("名字数量",len(words))

# print("longest name：",max(len(word) for word in words))
# print("shortest name：",min(len(word) for word in words))

# 统计出现的字符
chs=sorted(set("".join(words)))

# 字符与编号互相转化
stoi = {character: index+1 for index,character in enumerate(chs)}
stoi["."]=0
itos = {index: character for character,index in stoi.items()}

# 创建计数矩阵
N = torch.zeros((27,27),dtype=torch.int32)

# 填充计数矩阵
for word in words:
    characters=["."]+list(word)+["."]
    for ch1,ch2 in zip(characters,characters[1:]):
        ix1=stoi[ch1]
        ix2=stoi[ch2]
        N[ix1,ix2]+=1

# -------------------------------统计版本-----------------------------------

# 计算条件概率矩阵
P =(N+1).float()
P/=P.sum(1,keepdim=True)

# 生成器
generator = torch.Generator().manual_seed(2147483647)

# 生成名字
for _ in range(50):
    output=[]
    current_index=0

    while True:
        probabilities=P[current_index]
        # 根据概率分布采样下一个字符的索引multinomial(每个元素的概率分布，采样数量，是否放回采样，随机数生成器)
        next_index=torch.multinomial(probabilities,num_samples=1,replacement=True,generator=generator).item() 
        # 如果采样到的索引为0，表示生成结束
        if next_index==0:
            break
        output.append(itos[next_index])
        current_index=next_index

#    print("".join(output))



# -------------------------------神经网络版本-----------------------------------

# 构建神经网络训练数据
xs=[]
ys=[]

for word in words:
   characters=["."]+list(word)+["."]
   for ch1,ch2 in zip(characters,characters[1:]):
       ix1=stoi[ch1]
       ix2=stoi[ch2]

       xs.append(ix1)
       ys.append(ix2)

xs= torch.tensor(xs)
ys=torch.tensor(ys)

# 将数据转化为独热编码

xenc=F.one_hot(xs,num_classes=27).float()

# 创建神经网络权重
W=torch.randn((27,27),generator=generator,requires_grad=True)

# 训练神经网络
for step in range(100):
    # --------向前传播----------
    logits=xenc@W

    # 将logits转化为概率
    # 指数运算，保证每一个数字都是正数
    counts=logits.exp()

    # 对每一行进行归一化处理
    probabilities=counts/counts.sum(dim=1,keepdim=True)

    # 取出每条数据对正确字符给出的概率
    correct_probabilities=probabilities[torch.arange(xs.shape[0]),ys]

    # 计算平均负对数似然损失
    loss=-correct_probabilities.log().mean()

    # --------反向传播----------
    # 清空上次梯度
    W.grad=None

    # 计算当前损失对W的梯度
    loss.backward()

    # --------更新参数----------

    with torch.no_grad():
        W -=50*W.grad

    if step%10==0:
        print(step,loss.item())

# 利用训练好的模型生成名字
logits=xenc@W

# 将logits转化为概率
# 指数运算，保证每一个数字都是正数
counts=logits.exp()

# 对每一行进行归一化处理
probabilities=counts/counts.sum(dim=1,keepdim=True)

# 从 probabilities 中取出对应的27行
first_indices = torch.tensor([(xs == index).nonzero(as_tuple=True)[0][0].item()for index in range(27)])
probabilities_table = probabilities[first_indices]

# 生成的循环
for _ in range(50):

    output=[]
    current_index=0

    while True:
         next_index=torch.multinomial(probabilities_table[current_index],num_samples=1,replacement=True,generator=generator).item()        
         if next_index==0:
             break
         output.append(itos[next_index])

        # 把刚刚生成的字符作为下一次输入
         current_index=next_index

    print("".join(output))

