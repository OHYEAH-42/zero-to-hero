from dataclasses import dataclass,asdict
from pathlib import Path

import time
import torch
import math
import torch.nn as nn
from torch.nn import functional as F

# GPT超参数配置
@dataclass
class GPTConfig:
    block_size:int=1024
    vocab_size:int=50257
    n_layer:int=12
    n_head:int=12
    n_embd:int=768

# MLP多层感知器层
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.c_fc=nn.Linear(config.n_embd,4*config.n_embd)                                          # 线性层1
        self.gelu=nn.GELU(approximate="tanh")                                                       # 非线性层
        self.c_proj=nn.Linear(4*config.n_embd,config.n_embd)                                        # 线性层2
        self.c_proj.NANOGPT_SCALE_INIT=1

    def forward(self,x):

        x=self.c_fc(x)
        x=self.gelu(x)
        x=self.c_proj(x)

        return x

# 因果多头注意力
class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()

        assert config.n_embd % config.n_head==0

        self.n_head=config.n_head
        self.n_embd=config.n_embd

        self.c_attn=nn.Linear(self.n_embd,3*self.n_embd)                                            # 生成QKV的线性层
        self.c_proj=nn.Linear(self.n_embd,self.n_embd)                                              # 各头输出拼接后的线性变换
        self.c_proj.NANOGPT_SCALE_INIT = 1                                                          # 标记，这一层需要特殊初始化

    def forward(self,x):

        B,T,C=x.shape

        qkv=self.c_attn(x)                                                                          # 将qkv一起计算出来
        Q,K,V=qkv.split(self.n_embd,dim=-1)                                                         # qkv分开得到Q,K,V

        Q=Q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        K=K.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        V=V.view(B,T,self.n_head,C//self.n_head).transpose(1,2)

        # att=(Q@K.transpose(-1,-2))*(K.size(-1)**-0.5)                                               # 计算各个位置之间的匹配分数

        # mask=torch.tril(torch.ones(T,T,device=x.device,dtype=torch.bool))                           # 下三角为True,仅允许看到当前位置以前的部分
        # att=att.masked_fill(~mask,float("-inf"))                                                    # 屏蔽未来位置
        # att=F.softmax(att,dim=-1)                                                                   # 归一化

        # y=att@V
        # 替换为Flashattention
        y=F.scaled_dot_product_attention(Q,K,V,dropout_p=0.0,is_causal=True,)                         # SDPA是统一接口，有Pytorch自动选择注意力实现

        y=y.transpose(1,2).contiguous().view(B,T,C)                                                   # 把12个头拼接起来

        y=self.c_proj(y)                                                                              # 经过线性层
        return y

# Transformer block
class Block(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.ln_1=nn.LayerNorm(config.n_embd)                                                       # 归一化层
        self.attn=CausalSelfAttention(config)                                                       # 因果多头注意力层
        self.ln_2=nn.LayerNorm(config.n_embd)                                                       # 归一化层
        self.mlp=MLP(config)                                                                        # MLP层

    def forward(self,x):

        x=x+self.attn(self.ln_1(x))                                                                 # 注意力分支与残差连接
        x=x+self.mlp(self.ln_2(x))                                                                  # 多层感知器与残差连接
        return x

# GPT
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config=config

        self.transformer=nn.ModuleDict(dict(                                                        # Transformer
            wte=nn.Embedding(self.config.vocab_size,self.config.n_embd),                            # Token嵌入表
            wpe=nn.Embedding(self.config.block_size,self.config.n_embd),                            # 位置嵌入表
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),                        # 创建n_layer个Transformer Block
            ln_f=nn.LayerNorm(self.config.n_embd)))                                                 # 最后的归一化层

        self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias=False)                          # logits获取 

        self.transformer.wte.weight=self.lm_head.weight
        self.apply(self._init_weights)                                                              # apply递归遍历模型的各个模块，把他们逐个交给self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):                                                           # 判断当前模块是哪种层
            std = 0.02

            if hasattr(module, "NANOGPT_SCALE_INIT"):                                               # 检查它是否带有我们添加的标记
                std *= (2 * self.config.n_layer) ** -0.5

            nn.init.normal_(module.weight, mean=0.0, std=std)                                       # 直接修改原有参数中的数值

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self,idx,targets=None):
        B,T=idx.shape
        assert T <=self.config.block_size

        pos=torch.arange(T,device=idx.device)                                                       # 生成位置编号

        tok_emb=self.transformer.wte(idx)
        pos_emb=self.transformer.wpe(pos)
        x=tok_emb+pos_emb

        for block in self.transformer.h:
            x=block(x)

        x=self.transformer.ln_f(x)
        logits=self.lm_head(x)

        loss=None                                                                                   # 计算损失值
        if targets is not None:
            loss=F.cross_entropy(logits.reshape(B*T,self.config.vocab_size),targets.reshape(B*T))

        return logits,loss 
    @classmethod
    def from_pretrained(cls, model_type="gpt2", cache_dir=None):
        from transformers import GPT2LMHeadModel

        assert model_type == "gpt2"

        # 你的模型：等待接收权重
        model = cls(GPTConfig())
        sd = model.state_dict()

        # 预训练模型：提供已经训练好的权重
        model_hf = GPT2LMHeadModel.from_pretrained(
            model_type, cache_dir=cache_dir, use_safetensors=True
        )
        sd_hf = model_hf.state_dict()

        # 兼容部分版本中保存的注意力遮罩，排除这些非参数项
        sd_hf = {
            name: tensor
            for name, tensor in sd_hf.items()
            if not name.endswith((".attn.bias", ".attn.masked_bias"))
        }

        # 确认两边的名称一一对应
        assert set(sd.keys()) == set(sd_hf.keys())

        # 这些线性层的预训练权重存储方向与 nn.Linear 相反
        transposed = (
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        )

        with torch.no_grad():
            for name in sd:
                source = sd_hf[name]

                if name.endswith(transposed):
                    source = source.t()  # 将 [输入维度, 输出维度] 转为 [输出维度, 输入维度]

                assert sd[name].shape == source.shape, f"权重形状不匹配：{name}"

                sd[name].copy_(source)  # 原地复制，同时更新模型里的参数

        return model

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=40, temperature=1.0, top_k=50):
        """每次预测一个新 token，再把它接回输入，继续预测。"""
        if temperature <= 0:
            raise ValueError("temperature 必须大于 0")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k 必须大于 0，或设为 None")

        for _ in range(max_new_tokens):
            # 超过上下文长度时，只用最近的一段来预测
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # 只取最后位置的下一 token 分数

            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)

        return idx

# 数据加载器
class DataLoaderLite:
    def __init__(self, B, T, tokenizer,split="train"):
        self.B = B
        self.T = T
        self.current_position = 0

        # 读取文本，再编码成一条长长的 token 序列
        text = Path(__file__).with_name("input.txt").read_text(
            encoding="utf-8"
        )
        ids = tokenizer.encode(text)
        self.tokens = torch.tensor(ids, dtype=torch.long)

        n=int(len(self.tokens)*0.9)
        if split=="train":
            self.tokens=self.tokens[:n]
        else:
            self.tokens=self.tokens[n:]

        assert len(self.tokens) >= B * T + 1, "文本不足以构造一个批次"

    def next_batch(self):
        B, T = self.B, self.T

        # 剩余数据不够一个完整批次时，从头开始
        if self.current_position + B * T + 1 > len(self.tokens):
            self.current_position = 0

        # 从当前位置开始，取出连续的 B*T+1 个 token
        buf = self.tokens[self.current_position:self.current_position+B*T+1]

        # 前 B*T 个作为输入，后 B*T 个作为目标
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T)

        # 下一次调用时，从后面的数据继续取
        self.current_position += B * T

        return x, y

# 学习率获取
def get_lr(step,max_steps,warmup_steps=10,max_lr=1e-3,min_lr=1e-4):
    # 学习率预热：从较小学习率升到最高值
    if step<warmup_steps:
        return max_lr*(step+1)/warmup_steps

    # 超过调度终点后，保持最低学习率
    if step>max_steps:
        return min_lr

    # 在学习率最高值的时候，根据余弦函数，从最高值降到最低值
    progress=(step-warmup_steps)/(max_steps-warmup_steps)
    conefficient=0.5*(1+math.cos(math.pi*progress))

    return min_lr+conefficient*(max_lr-min_lr)

# 测试集loss检测
@torch.no_grad()
def evaluate_loss(model,loader,device,num_batches=10):
    model.eval()
    loader.current_position=0

    loss_total=torch.zeros((),device=device)

    for _ in range(num_batches):
        x,y=loader.next_batch()
        x,y=x.to(device),y.to(device)

        with torch.autocast(device_type="cuda",dtype=torch.bfloat16):                                           # 前向传播和损失计算使用自动混合精度
            _,loss=model(x,y)

        loss_total+=loss

    model.train()
    return (loss_total/num_batches).item()


if __name__ == "__main__":
    from transformers import GPT2Tokenizer

    # 首次运行自动下载；之后复用项目目录中的缓存
    cache_dir = str(Path(__file__).resolve().parent / ".hf-cache")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device=="cuda":
       torch.backends.cuda.matmul.fp32_precision="ieee"                # 矩阵乘法内部计算精度
    torch.manual_seed(42)

    print(f"使用设备：{device}", flush=True)
    print("正在读取 GPT-2 分词器，首次运行需要下载……", flush=True)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=cache_dir)

    # print("正在读取并复制 GPT-2 权重，首次运行需要下载……", flush=True)
    # model = GPT.from_pretrained("gpt2", cache_dir=cache_dir).to(device)
    # model.eval()

    # # 修改这里的文本，就可以让模型接着不同的开头写下去
    # prompt = "Hello, I'm a language model,"
    # token_ids = tokenizer.encode(prompt)
    # idx = torch.tensor([token_ids], dtype=torch.long, device=device)

    # print("权重加载完成，正在生成 40 个新 token……", flush=True)
    # output = model.generate(idx, max_new_tokens=40, temperature=1.0, top_k=50)
    # print("\n生成结果：")
    # print(tokenizer.decode(output[0].tolist(), skip_special_tokens=True))

    B, T = 2, 32
    loader = DataLoaderLite(B, T, tokenizer,split="train")
    val_loader=DataLoaderLite(B, T, tokenizer,split="val")

    # 创建随机初始化的小模型
    config = GPTConfig(
        block_size=T,
        n_layer=2,
        n_head=4,
        n_embd=64,
    )
    model = GPT(config).to(device)
    model.train()

    raw_model=model                                                                                                 # 编译前保留模型的引用，避免参数名称带上编译包装的前缀
    model=torch.compile(model)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # 进行权重衰减的优化器
    decay_params=[]                                                                                             
    no_decay_params=[]

    for p in model.parameters():
        if not p.requires_grad:                                                                                     # requires_grad:只把需要训练的参数交给优化器
            continue

        if p.dim()>=2:
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    param_groups=[                                                                                                  # 分别为两组设置衰减系数
        {"params":decay_params,"weight_decay":0.1},                                                                 # 0.1 不表示每次把权重减少 10%，实际缩小量还要乘学习率
        {"params":no_decay_params,"weight_decay":0.0}
    ]

    optimizer=torch.optim.AdamW(param_groups,lr=1e-3,betas=(0.9,0.95),eps=1e-8,fused=(device=="cuda"),)

    max_steps=100
    grad_accum_steps=4                                                                                              # 小批次的批次数，用以梯度累加，也就是多计算几次梯度才更新一次参数，相当于增加了batch数
    for step in range(max_steps):
        # 等待之前的GPU工作完成，再开始计时
        if device=="cuda":
            torch.cuda.synchronize()

        t0=time.perf_counter()                                                                                      # 开始时间计数

        # 每次参数更新前，清除上一轮梯度
        optimizer.zero_grad(set_to_none=True)
        loss_accum=torch.zeros((),device=device)

        for micro_step in range(grad_accum_steps):
            x, y = loader.next_batch()                                                                              # 加载x,y
            x, y = x.to(device), y.to(device)                                                                       # 写进所在设备的存储中

            # 前向计算，得到预测和损失
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):                                           # 前向传播和损失计算使用自动混合精度
                logits,loss=model(x,y)

            loss=loss/grad_accum_steps                                                                              # 让累计结果对应grad_accun_steps个批次的平均损失      
            loss_accum+=loss.detach()                                                                               # 记录损失，记录部分不参与求导                                                                         
            # 根据损失计算梯度
            loss.backward()                                                                                         # 每个小批次立即反向，梯度累加到参数上

        # 用梯度更新参数
        grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)                                   # 梯度裁剪，也就是限制梯度整体大小
        lr=get_lr(step,max_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"]=lr
        optimizer.step()

        # 等待这一轮GPU工作完成，再结束计时
        if device=="cuda":
            torch.cuda.synchronize()

        dt=time.perf_counter()-t0
        tokens_processed=B*T*grad_accum_steps
        tokens_per_second=tokens_processed/dt

        if step % 10 == 0:
            print(f"step {step}: loss = {loss_accum.item():.4f},"f"time={dt*1000:.1f}ms,"f"tokens/s={tokens_per_second:.1f},"f"grad_norm={grad_norm.item():.3f}, "f"lr={lr:.2e}, ")
        if (step + 1) % 10 == 0:
            val_loss = evaluate_loss(model, val_loader, device)
            print(f"验证 loss = {val_loss:.4f}")

    # 生成文本
    model.eval()

    prompt="First Citizen:"
    ids=tokenizer.encode(prompt)
    idx=torch.tensor([ids],dtype=torch.long,device=device)

    with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=(device=="cuda"),):
        output=model.generate(idx,max_new_tokens=80,temperature=0.8,top_k=50,)

    print("\n训练后的生成结果")
    print(tokenizer.decode(output[0].tolist()))

    # 参数保存
    save_path=Path(__file__).with_name("gpt2_shakespeare.pt")

    torch.save({"model":raw_model.state_dict(),"config":asdict(config),},save_path)
    print(f"模型已保存到{save_path}")

