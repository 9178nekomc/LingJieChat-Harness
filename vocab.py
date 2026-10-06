# vocab.py — 极简字符级+词级混合 tokenizer
import json
from config import (TOKENIZER_PATH, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN,
                    IM_START, IM_END, USER_TOKEN, ASSISTANT_TOKEN,
                    TOOL_START, TOOL_END, ARGS_START, ARGS_END)


CJK_PUNCT = "，。！？；：、“”‘’（）《》【】—…·"
EN_PUNCT = ".,!?;:\"'()[]{}<>-_+=*/\\|&^%$#@~`"


def is_cjk(ch):
    return '\u4e00' <= ch <= '\u9fff'


def split_text(text):
    """中文按字，英文按词，数字按字符，标点单独，特殊 token 整体。"""
    SPECIAL_TOKENS = ["<|im_start|>", "<|im_end|>", "<|user|>", "<|assistant|>",
                      "<tool>", "</tool>", "<args>", "</args>"]
    tokens = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue

        # 优先匹配特殊 token
        matched = False
        for st in SPECIAL_TOKENS:
            if text[i:i+len(st)] == st:
                tokens.append(st)
                i += len(st)
                matched = True
                break
        if matched:
            continue

        # 标点单独
        if is_cjk(c) or c in CJK_PUNCT or c in EN_PUNCT:
            tokens.append(c)
            i += 1
            continue

        # 数字逐字符
        if c.isdigit():
            tokens.append(c)
            i += 1
            continue

        # 英文按词
        j = i
        while j < n and not text[j].isspace() and not is_cjk(text[j]) \
              and text[j] not in EN_PUNCT and text[j] not in CJK_PUNCT \
              and not text[j].isdigit():
            # 提前检查是否是特殊 token 起点
            is_special = False
            for st in SPECIAL_TOKENS:
                if text[j:j+len(st)] == st:
                    is_special = True
                    break
            if is_special:
                break
            j += 1
        if j == i:
            j = i + 1
        tokens.append(text[i:j].lower())
        i = j
    return tokens


def detokenize(tokens):
    """把 token 序列拼回文本。规则：
    - 特殊 token 直接拼
    - 数字/小数点之间不加空格
    - 标点前后不加空格
    - 中文/英文单词之间加空格
    """
    if not tokens:
        return ""

    # 特殊 token 集合
    SPECIAL = set()
    for name in ["IM_START", "IM_END", "USER_TOKEN", "ASSISTANT_TOKEN",
                 "TOOL_START", "TOOL_END", "ARGS_START", "ARGS_END"]:
        if name in globals():
            SPECIAL.add(globals()[name])

    # 标点字符集合（不含反引号，避免转义问题）
    PUNCT = set(".,!?;:" + "\"'()[]{}<>|-_+=*/&^%$#@~")

    out = []
    prev = None

    for t in tokens:
        if t in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN):
            continue
        if t in SPECIAL:
            out.append(t)
            prev = None
            continue

        if prev is None:
            out.append(t)
        elif t.isdigit() and prev.isdigit():
            out.append(t)
        elif t == "." and prev.isdigit():
            out.append(t)
        elif prev == "." and t.isdigit():
            out.append(t)
        elif is_cjk(t) or is_cjk(prev):
            out.append(t)
        elif t in PUNCT:
            out.append(t)
        elif prev in PUNCT:
            # 标点后面：如果是句末标点，加空格；否则不加
            if prev in ".!?":
                out.append(" " + t)
            else:
                out.append(t)
        else:
            out.append(" " + t)
        prev = t

    return "".join(out).strip()


class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.stoi = {w: i for i, w in enumerate(vocab)}
        self.itos = {i: w for i, w in enumerate(vocab)}

    @classmethod
    def build(cls, texts):
        tokens = set()
        for t in texts:
            tokens.update(split_text(t))
        special = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN,
                   IM_START, IM_END, USER_TOKEN, ASSISTANT_TOKEN,
                   TOOL_START, TOOL_END, ARGS_START, ARGS_END]
        vocab = special + sorted(tokens)
        return cls(vocab)

    def encode(self, text):
        return [self.stoi.get(t, self.stoi[UNK_TOKEN]) for t in split_text(text)]

    def decode(self, ids):
        return detokenize([self.itos.get(i, UNK_TOKEN) for i in ids])

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            return cls(json.load(f))


if __name__ == "__main__":
    tok = Tokenizer.build(["你好 hello world，今天天气不错！"])
    print("Vocab size:", len(tok.vocab))
    ids = tok.encode("你好 hello world，今天天气不错！")
    print(ids)
    print(tok.decode(ids))
