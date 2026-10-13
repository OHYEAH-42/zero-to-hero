from pathlib import Path
from attention import Head,MultiHeadAttention,FeedForward,Block
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------读取文本与字符编码----------------------------------------------------------------------
# 读取数据
text=Path(__file__).with_name("input.txt").read_text(encoding="utf-8")

chars=sorted(set(text))                                                  # 字符集
vocab_size=len(chars)                                                    # 词汇表大小

stoi={ch:i for i,ch in enumerate(chars)}                                 # 字符到索引的映射
itos={i:ch for ch,i in stoi.items()}                                     # 索引到字符的映射

# 编码函数
def encode(text):
    return [stoi[ch]for ch in text]

# 解码函数
def decode(indices):
    return "".join([itos[i] for i in indices])

data = torch.tensor(encode(text),dtype=torch.long)                      # 将文本编码为索引的张量

# -----------------------------------------------------------------------------划分训练集和验证集--------------------------------------------------------------------
n=int(0.9*len(data))                                                    # 训练集大小
train_data=data[:n]                                                     # 训练集
val_data=data[n:]                                                       # 验证集                                       

block_size=128                                                            # 每个样本的长度
batch_size=12                                                            # 批量大小
n_embd=128                                                               # 特征向量维度
num_heads=4                                                             # 注意力头数量
n_layer=4                                                              # 使用的Transformer Block数
dropout=0.2                                                             # 随机失活参数
torch.manual_seed(1337)                                                 # 设置随机种子，保证每次运行结果一致

# -----------------------------------------------------------------------------构造训练批次--------------------------------------------------------------------------
# 随机截取多段连续文本，构造输入和向后错开一个字符的目标，形状均为 [batch_size, block_size]
def get_batch(spilt):
    source=train_data if spilt=="train" else val_data

    # 随机选取每段文本的起始位置
    starts =torch.randint(len(source)-block_size,(batch_size,))

    # 每段输入长度为 block_size；目标比输入向后错开一个字符，用于学习预测下一个字符
    x=torch.stack([source[i:i+block_size] for i in starts.tolist()])
    y=torch.stack([source[i+1:i+block_size+1] for i in starts.tolist()])

    return x,y

# -----------------------------------------------------------------------------语言模型-----------------------------------------------------------------------------
class SingleHeadLanguageModel(nn.Module):
    def __init__(self,vocab_size,n_embd,block_size):
        super().__init__()

        self.block_size=block_size
        # 将字符编号转化为特征向量
        self.token_embedding_table=nn.Embedding(vocab_size,n_embd)
        # 将位置编号转化为特征向量
        self.position_embedding_table=nn.Embedding(block_size,n_embd)

        self.sa=Head(n_embd=n_embd,head_size=n_embd,block_size=block_size)

        self.lm_head=nn.Linear(n_embd,vocab_size)

    def forward(self,idx,targets=None):
        B,T=idx.shape

        # 字符编号的特征向量 
        token_embeddings=self.token_embedding_table(idx)

        # 位置信息
        positions=torch.arange(T,device=idx.device)
        # 位置特征向量 
        position_embedding=self.position_embedding_table(positions)

        # 合并字符信息和位置信息，得到输入数据（B,T,C）
        x=token_embeddings+position_embedding

        x=self.sa(x)

        logits=self.lm_head(x)
        loss=None
        if targets is not None:
            B,T,V=logits.shape
            loss=F.cross_entropy(logits.reshape(B*T,V),targets.reshape(B*T))

        return logits,loss

    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            # 注意力遮罩和位置表最多支持block_size个位置
            idx_cond=idx[:,-self.block_size:]

            logits,_=self(idx_cond)
            next_logits=logits[:,-1,:]
            probs=F.softmax(next_logits,dim=-1)
            idx_next=torch.multinomial(probs,num_samples=1)

            # 保留完整生成文本，只预测时截取最近的上下文
            idx=torch.cat((idx,idx_next),dim=1)
        return idx

class MultiHeadLanguageModel(nn.Module):
    def __init__(self,vocab_size,n_embd,block_size,num_heads):
        super().__init__()

        self.block_size=block_size
        # 将字符编号转化为特征向量
        self.token_embedding_table=nn.Embedding(vocab_size,n_embd)
        # 将位置编号转化为特征向量
        self.position_embedding_table=nn.Embedding(block_size,n_embd)
        # 多头注意力
        self.sa=MultiHeadAttention(n_embd=n_embd,num_heads=num_heads,block_size=block_size)
        # 输出层
        self.lm_head=nn.Linear(n_embd,vocab_size)
        # 前馈网络
        self.ffwd=FeedForward(n_embd)

        # 归一化
        self.ln1=nn.LayerNorm(n_embd)
        self.ln2=nn.LayerNorm(n_embd)

    def forward(self,idx,targets=None):
        B,T=idx.shape

        # 字符编号的特征向量 
        token_embeddings=self.token_embedding_table(idx)

        # 位置信息
        positions=torch.arange(T,device=idx.device)
        # 位置特征向量 
        position_embedding=self.position_embedding_table(positions)

        # 合并字符信息和位置信息，得到输入数据（B,T,C）
        x=token_embeddings+position_embedding

        x=x+self.sa(self.ln1(x))                                    # 不同位置之间交流信息，attention的运作
        x=x+self.ffwd(self.ln2(x))                                  # 对每个位置的特征进一步加个，FeedForward的运作

        logits=self.lm_head(x)
        loss=None
        if targets is not None:
            B,T,V=logits.shape
            loss=F.cross_entropy(logits.reshape(B*T,V),targets.reshape(B*T))

        return logits,loss

    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            # 注意力遮罩和位置表最多支持block_size个位置
            idx_cond=idx[:,-self.block_size:]

            logits,_=self(idx_cond)
            next_logits=logits[:,-1,:]
            probs=F.softmax(next_logits,dim=-1)
            idx_next=torch.multinomial(probs,num_samples=1)

            # 保留完整生成文本，只预测时截取最近的上下文
            idx=torch.cat((idx,idx_next),dim=1)
        return idx

class GPTLanguageModel(nn.Module):
    def __init__(self,vocab_size,n_embd,block_size,num_heads,n_layer,dropout=0.0):
        super().__init__()

        self.block_size=block_size
        # 将字符编号转化为特征向量
        self.token_embedding_table=nn.Embedding(vocab_size,n_embd)
        # 将位置编号转化为特征向量
        self.position_embedding_table=nn.Embedding(block_size,n_embd)
        # 输出层 
        self.lm_head=nn.Linear(n_embd,vocab_size)

        self.blocks=nn.Sequential(*[Block(n_embd,num_heads,block_size,dropout=dropout) for _ in range(n_layer)])
        self.ln_f=nn.LayerNorm(n_embd)

    def forward(self,idx,targets=None):
        B,T=idx.shape

        # 字符编号的特征向量 
        token_embeddings=self.token_embedding_table(idx)

        # 位置信息
        positions=torch.arange(T,device=idx.device)
        # 位置特征向量 
        position_embedding=self.position_embedding_table(positions)

        # 合并字符信息和位置信息，得到输入数据（B,T,C）
        x=token_embeddings+position_embedding

        # 多头注意力、前馈网络处理
        x=self.blocks(x)
        # 归一化
        x=self.ln_f(x)

        logits=self.lm_head(x)
        loss=None
        if targets is not None:
            B,T,V=logits.shape
            loss=F.cross_entropy(logits.reshape(B*T,V),targets.reshape(B*T))

        return logits,loss

    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            # 注意力遮罩和位置表最多支持block_size个位置
            idx_cond=idx[:,-self.block_size:]

            logits,_=self(idx_cond)
            next_logits=logits[:,-1,:]
            probs=F.softmax(next_logits,dim=-1)
            idx_next=torch.multinomial(probs,num_samples=1)

            # 保留完整生成文本，只预测时截取最近的上下文
            idx=torch.cat((idx,idx_next),dim=1)
        return idx

class BigramLanguageModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()

        self.token_embedding_table=nn.Embedding(vocab_size,vocab_size) # 可学习的评分表：[当前字符编号, 下一字符候选编号]

    # 前向计算
    def forward(self,idx,targets=None):
        logits = self.token_embedding_table(idx) # 每个编号查出一行评分：[B, T] -> [B, T, V]

        loss=None # 没有提供目标时，只计算评分，损失保持为 None
        if targets is not None:
            B,T,V=logits.shape # B：批次大小；T：每段文本长度；V：词表大小

            # 将 B*T 个预测位置排在一起，用原始评分与目标编号计算平均交叉熵损失
            loss = F.cross_entropy(logits.reshape(B * T,V),targets.reshape(B*T))

        return logits,loss

    # 文本生成：idx 是初始文本的字符编号张量，形状为 [B, T]
    # max_new_tokens 是要新增的字符数量，不包含初始文本；本讲一个 token 对应一个字符
    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            logits,_=self(idx)
            next_logits=logits[:,-1,:]
            probs=F.softmax(next_logits,dim=1)
            idx_next=torch.multinomial(probs,num_samples=1)
            idx=torch.cat((idx,idx_next),dim=1)
        return idx
    
# -----------------------------------------------------------------------------模型初始化-----------------------------------------------------------------------------
xb,yb=get_batch("train")                                                                               # 输入，目标集合 
model=GPTLanguageModel(vocab_size,n_embd,block_size,num_heads,n_layer,dropout)                                   # 二元语言模型
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)                                                # AdamW优化器

# -----------------------------------------------------------------------------评估函数-----------------------------------------------------------------------------
def estimate_loss(spilt,eval_iters=100):
    was_training=model.training
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for _ in range(eval_iters):
            xb,yb=get_batch(spilt)
            _,loss=model(xb,yb)
            total_loss+=loss

    # 恢复评估之前的模式
    model.train(was_training)

    return total_loss/eval_iters
# -----------------------------------------------------------------------------模型训练-----------------------------------------------------------------------------
model.train()

best_val_loss = float("inf")  # 初始设为正无穷，第一次评估一定能刷新记录
save_path = Path(__file__).with_name("best_model.pt")

for step in range (20000):
    xb,yb=get_batch("train")
    logits,loss=model(xb,yb)

    # 清除上一轮梯度
    optimizer.zero_grad(set_to_none=True)

    # 根据当前损失计算梯度
    loss.backward()

    # 使用梯度更新参数
    optimizer.step()

    if step % 1000==0:
        train_loss=estimate_loss("train").item()
        val_loss=estimate_loss("val").item()

        print(f"step:{step+1} |"f"train:{train_loss:.4f}|"f"val:{val_loss:.4f}")
        
        # 验证集表现刷新记录，就保存当前模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print("已保存最佳模型")
# -----------------------------------------------------------------------------文本生成-----------------------------------------------------------------------------
# 将最佳参数载入当前模型，再进行生成
model.load_state_dict(torch.load(save_path, weights_only=True))

model.eval()
with torch.no_grad():
    idx=torch.tensor([encode("hello")],dtype=torch.long)
    output=model.generate(idx,200)

print (decode(output[0].tolist()))