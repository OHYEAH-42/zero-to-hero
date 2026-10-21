from pathlib import Path
import torch
from torch.nn import functional as F
from transformers import GPT2Tokenizer
from train_gpt2 import GPT, GPTConfig

# Hellaswag题目
example = {
    "ctx": "A man is sitting on a roof. he",
    "endings": [
        "is using wrap to wrap a pair of skis.",
        "is ripping level tiles off.",
        "is holding a rubik's cube.",
        "starts pulling up roofing on a roof.",
    ],
    "label": 3,
}

@torch.no_grad()
def score_ending(model,context, ending):
    context_ids = tokenizer.encode(context)
    ending_ids = tokenizer.encode(" " + ending)

    # 将上文与候选续写拼起来
    tokens = torch.tensor(
        [context_ids + ending_ids],
        dtype=torch.long,
        device=device,
    )

    # 每个位置预测下一个 token
    x = tokens[:, :-1]
    y = tokens[:, 1:]

    if x.size(1) > model.config.block_size:
        raise ValueError("题目过长，超过当前模型的上下文长度")

    logits, _ = model(x)

    # 保留每个位置的 loss，暂时不求平均
    losses = F.cross_entropy(
        logits[0],
        y[0],
        reduction="none",
    )

    # 只取候选续写对应的那些预测位置
    ending_losses = losses[len(context_ids) - 1:]

    return ending_losses.mean().item()

if __name__ == "__main__":
    # 脚本在子文件夹中，资源在项目根目录
    project_dir = Path(__file__).resolve().parent
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型
    checkpoint = torch.load(
        project_dir / "gpt2_shakespeare.pt",
        map_location="cpu",
        weights_only=True,
    )

    config = GPTConfig(**checkpoint["config"])
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    # 加载分词器
    tokenizer = GPT2Tokenizer.from_pretrained(
        "gpt2",
        cache_dir=str(project_dir.parent / ".hf-cache"),
        local_files_only=True,
    )

    # 分别计算四个选项的平均 loss
    scores = []

    for i, ending in enumerate(example["endings"]):
        score = score_ending(model, example["ctx"], ending)
        scores.append(score)
        print(f"选项 {i}: 平均 loss = {score:.4f}")

    prediction = torch.tensor(scores).argmin().item()

    print(f"模型选择：{prediction}")
    print(f"正确答案：{example['label']}")
    print(f"是否答对：{prediction == example['label']}")
