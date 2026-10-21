from pathlib import Path

import torch
from transformers import GPT2Tokenizer
from train_gpt2 import GPT,GPTConfig

device="cuda" if torch.cuda.is_available() else "cpu"
project_dir=Path(__file__).resolve().parent

# 读取保存的字典，先放在CPU上
checkpoint=torch.load(project_dir/"gpt2_shakespeare.pt",map_location="cpu",weights_only=True,)

# 创建相同结构的模型
config=GPTConfig(**checkpoint["config"])
model=GPT(config)

# 将参数装进模型
model.load_state_dict(checkpoint["model"])

# 移到运行设备，切换到评估模式
model=model.to(device)
model.eval()

# 读取之前缓存的GPT-2分词器
tokenizer=GPT2Tokenizer.from_pretrained("gpt2",cache_dir=str(project_dir/".hf-cache"),local_files_only=True)

# 将提示词编码成Token
prompt="First Citizen:"
ids=tokenizer.encode(prompt)
idx=torch.tensor([ids],dtype=torch.long,device=device)

# 使用加载的参数生成文本
with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=(device=="cuda"),):
    output=model.generate(idx,max_new_tokens=800,temperature=0.8,top_k=50)

print(tokenizer.decode(output[0].tolist()))
