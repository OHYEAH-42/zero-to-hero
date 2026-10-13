import torch
import torch.nn as nn
import torch.nn.functional as F

# 单头自注意力
class Head(nn.Module):
    # n_embd：每个 token 的输入特征维度
    # head_size：单个注意力头的 Q、K、V 特征维度
    # block_size：最大序列长度，用于构造因果注意力掩码
    # dropout: 让部分神经元失活，防止过拟合
    def __init__(self,n_embd,head_size,block_size,dropout=0.0):
        super().__init__()

        self.query=nn.Linear(n_embd,head_size,bias=False)
        self.key=nn.Linear(n_embd,head_size,bias=False)
        self.value=nn.Linear(n_embd,head_size,bias=False)

        self.head_size=head_size

        self.register_buffer("tril",torch.tril(torch.ones(block_size,block_size)))

        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        B,T,C=x.shape

        q=self.query(x)
        k=self.key(x)
        v=self.value(x)

        scores = q @ k.transpose(-2,-1)
        scores =scores*self.head_size**-0.5

        scores=scores.masked_fill(self.tril[:T,:T]==0,float("-inf"))

        weights = F.softmax(scores,dim=-1)
        weights = self.dropout(weights)

        return weights@v

# 多头自注意力
class MultiHeadAttention(nn.Module):
    def __init__(self,n_embd,num_heads,block_size,dropout=0.0):
        super().__init__()

        assert n_embd % num_heads == 0
        head_size=n_embd//num_heads

        # 每次调用Head()都创建一组独立参数
        self.heads=nn.ModuleList([Head(n_embd,head_size,block_size,dropout=dropout)for _ in range(num_heads)])

        # 混合拼接后各个头提供的信息
        self.proj=nn.Linear(n_embd,n_embd)

        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        outputs=[head(x) for head in self.heads]

        out=torch.cat(outputs,dim=-1) 

        return self.dropout(self.proj(out))

# 前馈神经网络
class FeedForward(nn.Module):
    def __init__(self,n_embd,dropout=0.0):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(n_embd,4*n_embd),nn.ReLU(),nn.Linear(4*n_embd,n_embd),nn.Dropout(dropout),)

    def forward(self,x):
        return self.net(x)

# Block
class Block(nn.Module):
    def __init__(self,n_embd,num_heads,block_size,dropout=0.0):
        super().__init__()

        self.sa=MultiHeadAttention(n_embd,num_heads,block_size,dropout=dropout)
        self.ffwd=FeedForward(n_embd,dropout)

        self.ln1=nn.LayerNorm(n_embd)
        self.ln2=nn.LayerNorm(n_embd)

    def forward(self,x):
        x=x+self.sa(self.ln1(x))
        x=x+self.ffwd(self.ln2(x))
        return x

if __name__=="__main__":
    torch.manual_seed(1337)

    x=torch.randn(2,4,32)
    head=Head(n_embd=32,head_size=16,block_size=8)

    out=head(x)
    print("输入形状：",x.shape)
    print("输出形状：",out.shape)