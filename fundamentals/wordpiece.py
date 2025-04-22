import re
import math
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional

class WordPieceTokenizer:
    def __init__(
        self,
        vocab_size: Optional[int] = None,
        num_merges: Optional[int] = None,
        unk_token: str = "[UNK]",
        pad_token: str = "[PAD]",
        end_word: str = "</w>",
    ):
        """
        Args:
          vocab_size: 最终词表大小，包括初始字符、合并出的子词和特殊符号
          num_merges:  合并轮数；如果指定了 vocab_size，则忽略此项，用 vocab_size - initial_size 来计算
          unk_token:   未登录词符号
          pad_token:   填充符号
          end_word:    单词结束标记
        """
        self.vocab_size = vocab_size
        self.num_merges = num_merges
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.end_word = end_word

        self.vocab: Counter = Counter()              # key: "s u b j e c t </w>", val: freq
        self.merges: List[Tuple[str,str]] = []        # 合并规则有序列表
        self.token2id: Dict[str,int] = {}             # 训练完毕后的 token → id 映射

    def _get_stats(self) -> Counter:
        """统计当前 vocab 中所有相邻对子 (x,y) 的加权出现次数"""
        pairs = Counter()
        for seq, freq in self.vocab.items():
            symbols = seq.split()
            for i in range(len(symbols)-1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def _merge_pair(self, pair: Tuple[str,str]) -> None:
        """在 self.vocab 上就地执行一次对子合并，更新 self.vocab"""
        a, b = pair
        pattern = re.compile(r'(?<!\S)' + re.escape(a+' '+b) + r'(?!\S)')
        new_vocab = Counter()
        for seq, freq in self.vocab.items():
            merged = pattern.sub(a+b, seq)
            new_vocab[merged] = freq
        self.vocab = new_vocab

    def fit(self, corpus: List[str]) -> None:
        """
        训练得到 merge 规则。
        corpus: 原始语料，按空格切分的“词”列表，例如 ["lower", "lowest", ...]
        """
        # 1) 构建初始 vocab：拆成字符 + end_word
        for word in corpus:
            key = ' '.join(list(word)) + ' ' + self.end_word
            self.vocab[key] += 1

        # 2) 决定合并轮数
        initial_tokens = set(sym for seq in self.vocab for sym in seq.split())
        special_count = {self.unk_token, self.pad_token}
        if self.vocab_size:
            target_merges = self.vocab_size - len(initial_tokens) - len(special_count)
            self.num_merges = max(0, target_merges)
        elif self.num_merges is None:
            raise ValueError("请指定 vocab_size 或 num_merges 其一")

        # 3) 迭代合并
        for _ in range(self.num_merges):
            pairs = self._get_stats()
            if not pairs:
                break

            # 3a) 计算 token 和 pair 的总 token 数
            token_counts = Counter()
            total_tokens = 0
            for seq, freq in self.vocab.items():
                toks = seq.split()
                total_tokens += len(toks) * freq
                token_counts.update({t: freq for t in toks})

            # 3b) 计算经验概率
            P_tok = {t: c/total_tokens for t,c in token_counts.items()}
            P_pair = {p: c/total_tokens for p,c in pairs.items()}

            # 3c) 计算似然增益 Δ = count * [log P(z) - log P(x) - log P(y)]
            gains = {
                pair: pairs[pair] * (
                    math.log(P_pair[pair])
                    - math.log(P_tok[pair[0]])
                    - math.log(P_tok[pair[1]])
                )
                for pair in pairs
            }

            # 3d) 选最优合并对并应用
            best = max(gains, key=gains.get)
            self.merges.append(best)
            self._merge_pair(best)

        # 4) 建立最终 token2id
        tokens = set()
        for seq in self.vocab:
            tokens.update(seq.split())
        # 加入特殊符号
        tokens |= {self.unk_token, self.pad_token}
        # 按出现顺序：先初始字符，再合并生成，再 special
        ordered = list(tokens - special_count) + [self.unk_token, self.pad_token]
        self.token2id = {tok: i for i,tok in enumerate(ordered)}

    def encode_word(self, word: str) -> List[str]:
        """
        对单个词做最长匹配分词，输出子词列表
        """
        # 拆成字符 + end_word
        symbols = list(word) + [self.end_word]
        i = 0
        # 贪心：不断应用合并规则
        for a,b in self.merges:
            j = 0
            while j < len(symbols)-1:
                if symbols[j] == a and symbols[j+1] == b:
                    symbols[j:j+2] = [a+b]
                else:
                    j += 1
        # 最长匹配切分（从左扫描，取最长 token）
        output = []
        s = ''.join(symbols)
        while s:
            # 从最长可能 token 开始找
            for length in range(len(s), 0, -1):
                piece = s[:length]
                if piece in self.token2id:
                    output.append(piece)
                    s = s[length:]
                    break
            else:
                # 没匹配到，退化到单字符 or unk
                output.append(self.unk_token)
                s = s[1:] if len(s)>1 else ''
        return output

    def encode(self, text: str) -> List[str]:
        """
        对一句话做分词，先按空格切词，再对每个词调用 encode_word
        """
        tokens = []
        for w in text.strip().split():
            tokens.extend(self.encode_word(w))
        return tokens

# -----------------------
# 使用示例
# -----------------------
if __name__ == "__main__":
    # 1) 训练语料：一系列词（也可根据需要对句子再做 split）
    corpus = ["low", "lowest", "newer", "wider", "low", "newest", "wide"]

    # 2) 初始化并训练到 50 个合并操作
    wp = WordPieceTokenizer(num_merges=50)
    wp.fit(corpus)

    print("合并规则：")
    for i,(a,b) in enumerate(wp.merges,1):
        print(f"{i:2d}. {a}+{b}→{a+b}")

    # 3) 编码测试
    for w in ["lowest", "newest", "wider", "unknown"]:
        pieces = wp.encode_word(w)
        print(f"{w:>8} → {pieces}")
    print("句子示例：", wp.encode("low lowest wide"))
