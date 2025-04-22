import math
from collections import defaultdict, Counter
from typing import List, Dict, Set

def get_initial_vocab(corpus: List[str], max_token_length: int = 5) -> Set[str]:
    # 构建初始候选词表：所有子串，长度 1..max_token_length
    vocab = set()
    for word in corpus:
        L = len(word)
        for i in range(L):
            for l in range(1, min(max_token_length, L - i) + 1):
                vocab.add(word[i:i+l])
    return vocab


def viterbi_segment(word: str, token_probs: Dict[str, float], max_token_length: int) -> List[str]:
    # Viterbi 分词，按 unigram 概率寻找最优切分
    n = len(word)
    dp = [-1e9] * (n + 1)
    back = [0] * (n + 1)
    dp[0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(0, i - max_token_length), i):
            piece = word[j:i]
            if piece in token_probs:
                score = dp[j] + math.log(token_probs[piece])
                if score > dp[i]:
                    dp[i] = score
                    back[i] = j
    # 回溯
    tokens = []
    i = n
    while i > 0:
        j = back[i]
        tokens.insert(0, word[j:i])
        i = j
    return tokens


def train_unigram(
    corpus: List[str],
    vocab_size: int,
    max_token_length: int = 5,
    em_iterations: int = 5,
) -> Dict[str, float]:
    # 1) 初始词表
    init_tokens = get_initial_vocab(corpus, max_token_length)
    token_probs = {t: 1.0 / len(init_tokens) for t in init_tokens}

    # 2) EM 迭代
    for _ in range(em_iterations):
        counts = defaultdict(float)
        total = 0.0
        for word in corpus:
            segs = viterbi_segment(word, token_probs, max_token_length)
            for t in segs:
                counts[t] += 1.0
                total += 1.0
        for t, c in counts.items():
            token_probs[t] = c / total

    # 3) 剪枝到目标大小
    while len(token_probs) > vocab_size:
        low = min(token_probs, key=token_probs.get)
        del token_probs[low]
    # 归一化
    Z = sum(token_probs.values())
    for t in token_probs:
        token_probs[t] /= Z
    return token_probs


if __name__ == '__main__':
    corpus = ['low', 'lowest', 'newer', 'wider', 'low', 'newest', 'wide']
    token_probs = train_unigram(corpus, vocab_size=50)
    print('Final vocab size:', len(token_probs))
    print('Sample tokens:', list(token_probs.items())[:10])

    for w in ['lowest', 'newest', 'unknown']:
        seg = viterbi_segment(w, token_probs, max_token_length=5)
        print(f'{w} -> {seg}')
