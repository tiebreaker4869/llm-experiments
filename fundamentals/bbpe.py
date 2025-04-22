from collections import defaultdict, Counter
from typing import List, Tuple

class ByteLevelBPETokenizer:
    def __init__(self, num_merges: int):
        """
        Args:
          num_merges: 要执行的合并轮数
        """
        self.num_merges = num_merges
        self.bpe_codes: List[Tuple[bytes, bytes]] = []  # 记录每轮合并的字节对子
        self.vocab: Counter = Counter()                 # 初始字节序列频次

    def get_stats(self, vocab: Counter) -> Counter:
        """
        统计所有相邻字节对子出现的加权频次
        vocab 的 key 是 tuple(bytes), value 是频次
        """
        pairs = Counter()
        for seq, freq in vocab.items():
            for i in range(len(seq) - 1):
                pairs[(seq[i], seq[i+1])] += freq
        return pairs

    def merge_pair(self, pair: Tuple[bytes, bytes], vocab_in: Counter) -> Counter:
        """
        在所有序列里，将 pair 合并成一个新 token（bytes 拼接）
        """
        a, b = pair
        new_vocab = Counter()
        for seq, freq in vocab_in.items():
            merged = []
            i = 0
            while i < len(seq):
                # 如果当前位置正好是我们要合并的 pair，就把它们拼在一起
                if i < len(seq)-1 and seq[i] == a and seq[i+1] == b:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(seq[i:i+1])
                    i += 1
            new_vocab[tuple(merged)] += freq
        return new_vocab

    def fit(self, corpus: List[str]):
        """
        基于输入语料训练合并规则
        corpus: 每个元素是一段文本（可包含空格、标点等）
        """
        # 1) 构建初始 vocab：按 UTF-8 编码拆成字节序列
        for text in corpus:
            bseq = tuple(text.encode("utf-8"))
            self.vocab[bseq] += 1

        # 2) 迭代合并
        for _ in range(self.num_merges):
            pairs = self.get_stats(self.vocab)
            if not pairs:
                break
            best_pair = pairs.most_common(1)[0][0]
            self.bpe_codes.append(best_pair)
            self.vocab = self.merge_pair(best_pair, self.vocab)

    def encode(self, text: str) -> List[bytes]:
        """
        用训练好的 bpe_codes 对新文本进行分词（字节级）
        返回一个字节序列列表，每一项是合并后的子词（bytes）
        """
        seq = [bytes([b]) for b in text.encode("utf-8")]
        for a, b in self.bpe_codes:
            i = 0
            while i < len(seq)-1:
                if seq[i] == a and seq[i+1] == b:
                    seq[i:i+2] = [a + b]
                else:
                    i += 1
        return seq

# -----------------------
# 使用示例
# -----------------------
if __name__ == "__main__":
    corpus = [
        "Hello, world!",
        "Hi there!",
        "Hello, BPE!"
    ]
    tokenizer = ByteLevelBPETokenizer(num_merges=50)
    tokenizer.fit(corpus)

    print("Learned BPE merges (in byte‐pairs):")
    for i, (a, b) in enumerate(tokenizer.bpe_codes, 1):
        print(f"{i}: {a!r} + {b!r} → {a+b!r}")

    sample = "Hello!"
    encoded = tokenizer.encode(sample)
    print(f"\nOriginal bytes: {list(sample.encode('utf-8'))}")
    print("BPE tokens:", [tok for tok in encoded])
