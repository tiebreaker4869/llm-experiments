import re
from collections import defaultdict, Counter
from typing import List, Tuple, Dict

class BytePairEncoding:
    def __init__(self, num_merges: int):
        """
        Args:
          num_merges: 要执行的合并轮数（最终子词表规模 ≈ init_vocab_size + num_merges）
        """
        self.num_merges = num_merges
        self.bpe_codes: List[Tuple[str, str]] = []  # 存储每轮合并产生的对子 (a, b)
        self.vocab: Dict[str, int] = {}           # 初始化时的 word→freq 字典

    @staticmethod
    def get_stats(vocab: Dict[str,int]) -> Dict[Tuple[str,str], int]:
        """
        统计所有相邻 token 对出现的频次。
        vocab 的 key 形如 "s u b j e c t </w>"
        """
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    @staticmethod
    def merge_pair(pair: Tuple[str,str], vocab_in: Dict[str,int]) -> Dict[str,int]:
        """
        在所有词上将 pair 合并成一个新符号。
        如 pair=('s','u') 会把 "s u b j e c t </w>" → "su b j e c t </w>"
        """
        vocab_out = {}
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word, freq in vocab_in.items():
            # 用正则将所有匹配的 pair 合并
            new_word = pattern.sub(''.join(pair), word)
            vocab_out[new_word] = freq
        return vocab_out

    def fit(self, corpus: List[str]):
        """
        训练 BPE：先构建初始 vocab，再执行 num_merges 轮合并。
        corpus: 每个元素都是一个原始单词（string）
        """
        # 1) 初始化 vocab：拆成字符 + </w>
        self.vocab = Counter(' '.join(list(word)) + ' </w>' for word in corpus)

        # 2) 迭代合并
        for i in range(self.num_merges):
            pairs = self.get_stats(self.vocab)
            if not pairs:
                break
            # 找到最频繁的 pair
            best_pair = max(pairs, key=pairs.get)
            self.bpe_codes.append(best_pair)
            # 合并它
            self.vocab = self.merge_pair(best_pair, self.vocab)

    def encode(self, word: str) -> List[str]:
        """
        用学到的 bpe_codes 将一个新单词编码成子词序列。
        """
        # 初始：拆成字符 + </w>
        symbols = list(word) + ['</w>']
        for a, b in self.bpe_codes:
            i = 0
            # 对每轮合并检查整个符号列表
            while i < len(symbols)-1:
                if symbols[i] == a and symbols[i+1] == b:
                    symbols[i:i+2] = [a+b]  # 合并
                else:
                    i += 1
        return symbols
    

if __name__ == "__main__":
    # 1) 训练语料：简单示例
    corpus = [
        "low", "lowest", "newer", "wider", "low", "newest", "wide"
    ]
    bpe = BytePairEncoding(num_merges=10)
    bpe.fit(corpus)

    print("Learned merges:")
    for i, pair in enumerate(bpe.bpe_codes, 1):
        print(f"  {i}. {pair}")

    # 2) 用训练好的规则编码新词
    test_words = ["lowest", "newest", "wider"]
    for w in test_words:
        print(f"{w} → {bpe.encode(w)}")