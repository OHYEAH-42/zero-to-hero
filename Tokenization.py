"""P9：从零实现字节级 BPE Tokenizer。

实现顺序：get_stats → merge → train → encode → decode。
基础 token ID 为 0～255，新增 token ID 从 256 开始。
"""

# 传入ids字符串，返回token对以及对应的次数
def get_stats(ids):
    """统计相邻 token 对的次数，返回 {(左 ID, 右 ID): 次数}。"""
    counts = {}
    for pair in zip(ids,ids[1:]):
        counts[pair]=counts.get(pair,0)+1
    return counts

# ids字符串，pair从get_stats得到的字典中的一对pair，new_id为新增加的id
def merge(ids, pair, new_id):
    """从左到右将 pair 的非重叠出现替换成 new_id，返回新列表。"""
    # TODO：遇到目标 pair 时前进两个位置，否则保留当前 ID 并前进一个位置。
    new_ids=[]
    i=0
    while i <len(ids):
        if i+1<len(ids) and (ids[i],ids[i+1])==pair:
            new_ids.append(new_id)
            i+=2
        else:
            new_ids.append(ids[i])
            i+=1
    return new_ids
        
def train(text, num_merges):
    """学习最多 num_merges 次合并，返回 merges 和 vocab。

    merges：{(左 ID, 右 ID): 新 ID}，新 ID 的大小也表示合并先后顺序。
    vocab：{token ID: 对应的 bytes}，包括基础的 256 个字节 token。
    """
    text_tran=text.encode("utf-8")
    ids=list(text_tran)
    merges={}
    vocab={}
    for i in range(256):
        vocab[i]=bytes([i])
    for i in range(num_merges):
        counts=get_stats(ids)
        if not counts:
            break
        pair=max(counts,key=counts.get)
        ids=merge(ids,pair,256+i)
        merges[pair]=256+i
        vocab[256+i] = vocab[pair[0]] + vocab[pair[1]]
    return merges,vocab



def encode(text, merges):
    """用已训练的规则将新文本编码为 token ID 列表。"""
    # TODO：先将 text 转为 UTF-8 字节的整数列表。
    ids=list(text.encode("utf-8"))
    # TODO：在当前相邻对中，选择训练时最早学到的可用规则并合并。
    while True:
        counts=get_stats(ids)
        min_id = float("inf")
        min_pair=None
        for pair in counts:
            if pair in merges and merges[pair]<min_id:
                min_id = merges[pair]
                min_pair=pair
        if min_pair==None:
            return ids
        ids=merge(ids,min_pair,min_id)


def decode(ids, vocab):
    """将每个 token 的字节串拼接起来，再整体按 UTF-8 解码。"""
    # TODO：单个 token 不一定是完整字符，必须拼接字节后再解码。
    text =b""
    for i in range(len(ids)):
        text+=vocab[ids[i]]
    result=text.decode("utf-8")
    return result
        

if __name__ == "__main__":
    # 先实现 get_stats，再取消下面两行的注释，检查统计结果。
    # ids = [1, 2, 1, 2, 3]
    # print(get_stats(ids))
    pass
